# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from manydepth import datasets, networks
from manydepth.layers import (SSIM, BackprojectDepth, Project3D,
                              compute_depth_errors, disp_to_depth,
                              get_smooth_loss, transformation_from_parameters)
from manydepth.options import MonodepthOptions
from manydepth.trainer import Trainer as ManydepthTrainer
from manydepth.trainer import seed_worker
from manydepth.utils import readlines, sec_to_hm_str
from loguru import logger


def count_trainable_params(model: nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        if p.requires_grad_:
            trainable += p.numel()
        total += p.numel()
    logger.info('Num of trainable: {}/ {} ({:0.2f})%'.format(trainable,
                total, 100*trainable/total))


def get_trainer(exp_name, gpus=1, max_epochs=40,
                distributed=False, trainer_strategy='dp',
                monitor=dict(metric="loss_val", mode="min")
                ):
    import os.path as osp
    from datetime import datetime, timedelta

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import (LearningRateMonitor,
                                             ModelCheckpoint, TQDMProgressBar)
    from pytorch_lightning.loggers import TensorBoardLogger

    now = datetime.now() + timedelta(hours=7)
    root_log_dir = osp.join(
        "lightning_logs", exp_name, now.strftime(
            "%b%d-%H:%M:%S")
    )

    callback_ckpt = ModelCheckpoint(
        dirpath=osp.join(root_log_dir, "ckpts"),
        monitor=monitor['metric'],
        filename="{epoch}-{loss_val:.2f}",
        mode=monitor['mode'],
        save_last=True
    )

    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    plt_logger = TensorBoardLogger(
        os.path.join(root_log_dir, "tb_logs"), version=now.strftime("%b%d-%H:%M:%S")
    )
    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        strategy=trainer_strategy if distributed else 'dp',
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=plt_logger,
    )
    return trainer


def prepare_model(opt, num_pose_frames):
    models = nn.ModuleDict()
    models["encoder"] = networks.ResnetEncoderMatching(
        opt.num_layers, opt.weights_init == "pretrained",
        input_height=opt.height, input_width=opt.width,
        adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
        depth_binning=opt.depth_binning, num_depth_bins=opt.num_depth_bins)

    models["depth"] = networks.DepthDecoder(
        models["encoder"].num_ch_enc, opt.scales)

    models["mono_encoder"] = \
        networks.ResnetEncoder(18, opt.weights_init == "pretrained")

    models["mono_depth"] = \
        networks.DepthDecoder(
            models["mono_encoder"].num_ch_enc, opt.scales)

    models["pose_encoder"] = \
        networks.ResnetEncoder(18, opt.weights_init == "pretrained",
                               num_input_images=num_pose_frames)

    models["pose"] = \
        networks.PoseDecoder(models["pose_encoder"].num_ch_enc,
                             num_input_features=1,
                             num_frames_to_predict_for=2)
    return models


class LitModel(LightningModule):
    def __init__(self, options, matching_ids):
        super().__init__()
        self.opt = options
        self.matching_ids = matching_ids
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = nn.ModuleDict()
        self.parameters_to_train = []
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(
            self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            print('using adaptive depth binning!')
            self.min_depth_tracker = nn.Parameter(torch.Tensor([0.1]), False)
            self.max_depth_tracker = nn.Parameter(torch.Tensor([10.0]), False)
        else:
            print('fixing pose network and monocular network!')

        # MODEL SETUP

        self.models = prepare_model(self.opt, self.num_pose_frames)

        # if self.opt.load_weights_folder is not None:
        #     self.load_model()

        # if self.opt.mono_weights_folder is not None:
        #     self.load_mono_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ",
              self.opt.log_dir)


        if not self.opt.no_ssim:
            self.ssim = SSIM()

        self.backproject_depth = nn.ModuleDict()
        self.project_3d = nn.ModuleDict()

        self.opt.scales = [str(s) for s in self.opt.scales]

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** int(scale))
            w = self.opt.width // (2 ** int(scale))

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale]

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        # print("There are {:d} training items and {:d} validation items\n".format(
        #     len(train_dataset), len(val_dataset)))

        self.save_opts()

    def configure_optimizers(self):
        model_optimizer = optim.Adam(
            self.models.parameters(), self.opt.learning_rate)
        model_lr_scheduler = optim.lr_scheduler.StepLR(
            model_optimizer, self.opt.scheduler_step_size, 0.1)

        return [model_optimizer], [model_lr_scheduler]

    # def set_train(self):
    #     """Convert all models to training mode
    #     """

    #     for k, m in self.models.items():
    #         if self.train_teacher_and_pose:
    #             m.train()
    #         else:
    #             if k in ['depth', 'encoder']:
    #                 m.train()

    # def set_eval(self):
    #     """Convert all models to testing/evaluation mode
    #     """
    #     for m in self.models.values():
    #         m.eval()
    def on_epoch_start(self) -> None:
        if self.current_epoch == self.opt.freeze_teacher_epoch:
            self.freeze_teacher()
        return super().on_epoch_start()
    # def train(self):
    #     """Run the entire training pipeline
    #     """
    #     self.epoch = 0
    #     self.step = 0
    #     self.start_time = time.time()
    #     for self.epoch in range(self.opt.num_epochs):
    #         if self.epoch == self.opt.freeze_teacher_epoch:
    #             self.freeze_teacher()

    #         self.run_epoch()
    #         if (self.epoch + 1) % self.opt.save_frequency == 0:
    #             self.save_model()

    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            logger.info('freezing teacher and pose networks!: ')
            # count_trainable_params(self.models)
            for model_name in self.models.keys():
                if model_name in ['encoder', 'depth']:
                    logger.info('Train ', model_name)
                    for param in self.models[model_name].parameters():
                        param.requires_grad_ = True
                else:
                    logger.info('Frozen', model_name)
                    for param in self.models[model_name].parameters():
                        param.requires_grad_ = False
            count_trainable_params(self.models)

    # def run_epoch(self):
    #     """Run a single epoch of training and validation
    #     """

    #     print("Training")
    #     self.set_train()

    #     for batch_idx, inputs in enumerate(self.train_loader):

    #         before_op_time = time.time()

    #         outputs, losses = self.process_batch(inputs, is_train=True)
    #         self.model_optimizer.zero_grad()
    #         losses["loss"].backward()
    #         self.model_optimizer.step()

    #         duration = time.time() - before_op_time

    #         # log less frequently after the first 2000 steps to save time & disk space
    #         early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
    #         late_phase = self.step % 2000 == 0

    #         if early_phase or late_phase:
    #             self.log_time(batch_idx, duration, losses["loss"].cpu().data)

    #             if "depth_gt" in inputs:
    #                 self.compute_depth_losses(inputs, outputs, losses)

    #             self.log("train", inputs, outputs, losses)
    #             self.val()

    #         if self.opt.save_intermediate_models and late_phase:
    #             self.save_model(save_step=True)

    #         if self.step == self.opt.freeze_teacher_step:
    #             self.freeze_teacher()

    #         self.step += 1
    #     self.model_lr_scheduler.step()

    def training_step(self, inputs, _):
        """Pass a minibatch through the network and generate images and losses
        """
        if not self.train_teacher_and_pose:
            for m in self.models.keys():
                if not m in ['depth', 'encoder']:
                    self.models[m].eval()

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, None)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)]
                          for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)]
                         for idx in self.matching_ids[1:]]
        # batch x frames x 3 x h x w
        lookup_frames = torch.stack(lookup_frames, 1)

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1])

        if self.training and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx]
                         for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models['mono_depth'](feats))

        self.generate_images_pred(inputs, mono_outputs)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[(
                                                                            'K', 2)],
                                                                        inputs[(
                                                                            'inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin)
        outputs.update(self.models["depth"](features))
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.opt.height, self.opt.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height,
                                                        self.opt.width],
                                                    mode="nearest")[:, 0]

        if not self.opt.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                           self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val

        # update adaptive depth bins
        if self.train_teacher_and_pose:
            self.update_adaptive_depth_bins(outputs)

        for k, v in losses.items():
            metric_name = '{}_{}'.format(
                k, 'training' if self.training else 'val')
            self.log(metric_name, v, prog_bar=True, rank_zero_only=True)

        return losses['loss']

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)
                            ].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)
                            ].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean()#.cpu().item()
        max_depth = max_depth.mean()#.cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = nn.Parameter(self.max_depth_tracker * 0.99 + max_depth * 0.01, False)
        self.min_depth_tracker = nn.Parameter(self.min_depth_tracker * 0.99 + min_depth * 0.01, False)

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0]
                          for f_i in self.opt.frame_ids}
            for frame_index in self.opt.frame_ids[1:]:
                if frame_index != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if frame_index < 0:
                        pose_inputs = [pose_feats[frame_index], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[frame_index]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, frame_index)] = axisangle
                    outputs[("translation", 0, frame_index)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, frame_index)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(frame_index < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {frame_index: inputs["color_aug", frame_index, 0]
                          for frame_index in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](
                            torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](
                            pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(
                                pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](
                            torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](
                            pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(
                                pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    # def val(self):
    #     """Validate the model on a single minibatch
    #     """
    #     self.set_eval()
    #     try:
    #         inputs = self.val_iter.next()
    #     except StopIteration:
    #         self.val_iter = iter(self.val_loader)
    #         inputs = self.val_iter.next()

    #     with torch.no_grad():
    #         outputs, losses = self.process_batch(inputs)

    #         if "depth_gt" in inputs:
    #             self.compute_depth_losses(inputs, outputs, losses)

    #         self.log("val", inputs, outputs, losses)
    #         del inputs, outputs, losses

    #     self.set_train()

    def validation_step(self, batch, _):
        return self.training_step(batch, None)

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            scale = int(scale)
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(
                disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)] # 4x4
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[str(source_scale)](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[str(source_scale)](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat(
                [reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / \
            outputs['lowest_cost'].unsqueeze(1).to(mono_output.device)

        # mask where they differ by a large amount

        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            scale = int(scale)
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(
                    identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(
                        1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(
                    reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn_like(
                    identity_reprojection_loss) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(
                    reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask'].to(reprojection_loss_mask.device)))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(
                    multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)
                        ] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    # def log(self, mode, inputs, outputs, losses):
    #     """Write an event to the tensorboard events file
    #     """
    #     writer = self.writers[mode]
    #     for l, v in losses.items():
    #         writer.add_scalar("{}".format(l), v, self.step)

    #     for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
    #         s = 0  # log only max scale
    #         for frame_id in self.opt.frame_ids:
    #             writer.add_image(
    #                 "color_{}_{}/{}".format(frame_id, s, j),
    #                 inputs[("color", frame_id, s)][j].data, self.step)
    #             if s == 0 and frame_id != 0:
    #                 writer.add_image(
    #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
    #                     outputs[("color", frame_id, s)][j].data, self.step)

    #         disp = colormap(outputs[("disp", s)][j, 0])
    #         writer.add_image(
    #             "disp_multi_{}/{}".format(s, j),
    #             disp, self.step)

    #         disp = colormap(outputs[('mono_disp', s)][j, 0])
    #         writer.add_image(
    #             "disp_mono/{}".format(j),
    #             disp, self.step)

    #         if outputs.get("lowest_cost") is not None:
    #             lowest_cost = outputs["lowest_cost"][j]

    #             consistency_mask = \
    #                 outputs['consistency_mask'][j].cpu(
    #                 ).detach().unsqueeze(0).numpy()

    #             min_val = np.percentile(lowest_cost.numpy(), 10)
    #             max_val = np.percentile(lowest_cost.numpy(), 90)
    #             lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
    #             lowest_cost = colormap(lowest_cost)

    #             writer.add_image(
    #                 "lowest_cost/{}".format(j),
    #                 lowest_cost, self.step)
    #             writer.add_image(
    #                 "lowest_cost_masked/{}".format(j),
    #                 lowest_cost * consistency_mask, self.step)
    #             writer.add_image(
    #                 "consistency_mask/{}".format(j),
    #                 consistency_mask, self.step)

    #             consistency_target = colormap(
    #                 outputs["consistency_target/0"][j])
    #             writer.add_image(
    #                 "consistency_target/{}".format(j),
    #                 consistency_target, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(
                self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    # def load_mono_model(self):
    #     model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
    #     for n in model_list:
    #         print('loading {}'.format(n))
    #         path = os.path.join(
    #             self.opt.mono_weights_folder, "{}.pth".format(n))
    #         model_dict = self.models[n].state_dict()
    #         pretrained_dict = torch.load(path)

    #         pretrained_dict = {k: v for k,
    #                            v in pretrained_dict.items() if k in model_dict}
    #         model_dict.update(pretrained_dict)
    #         self.models[n].load_state_dict(model_dict)

    # def load_model(self):
    #     """Load model(s) from disk
    #     """
    #     self.opt.load_weights_folder = os.path.expanduser(
    #         self.opt.load_weights_folder)

    #     assert os.path.isdir(self.opt.load_weights_folder), \
    #         "Cannot find folder {}".format(self.opt.load_weights_folder)
    #     print("loading model from folder {}".format(
    #         self.opt.load_weights_folder))

    #     for n in self.opt.models_to_load:
    #         print("Loading {} weights...".format(n))
    #         path = os.path.join(
    #             self.opt.load_weights_folder, "{}.pth".format(n))
    #         model_dict = self.models[n].state_dict()
    #         pretrained_dict = torch.load(path)

    #         if n == 'encoder':
    #             min_depth_bin = pretrained_dict.get('min_depth_bin')
    #             max_depth_bin = pretrained_dict.get('max_depth_bin')
    #             print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
    #             if min_depth_bin is not None:
    #                 # recompute bins
    #                 print('setting depth bins!')
    #                 self.models['encoder'].compute_depth_bins(
    #                     min_depth_bin, max_depth_bin)

    #                 self.min_depth_tracker = min_depth_bin
    #                 self.max_depth_tracker = max_depth_bin

    #         pretrained_dict = {k: v for k,
    #                            v in pretrained_dict.items() if k in model_dict}
    #         model_dict.update(pretrained_dict)
    #         self.models[n].load_state_dict(model_dict)

    #     # loading adam state
    #     optimizer_load_path = os.path.join(
    #         self.opt.load_weights_folder, "adam.pth")
    #     if os.path.isfile(optimizer_load_path):
    #         try:
    #             print("Loading Adam weights")
    #             optimizer_dict = torch.load(optimizer_load_path)
    #             self.model_optimizer.load_state_dict(optimizer_dict)
    #         except ValueError:
    #             print("Can't load Adam - using random")
    #     else:
    #         print("Cannot find Adam weights so Adam is randomly initialized")


def get_num_gpus(gpus):
    if isinstance(gpus, list):
        return len(gpus)
    elif isinstance(gpus, int):
        return gpus
    else:
        raise Exception('{} is not implemented'.format(type(gpus)))


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    gpus = opts.gpus
    lit_trainer = get_trainer(opts.model_name,
                              gpus=gpus, trainer_strategy='ddp')

    # DATA
    fpath = os.path.join("splits", opts.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png' if opts.png else '.jpg'

    num_train_samples = len(train_filenames)
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                     "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                     "kitti_odom": datasets.KITTIOdomDataset}
    dataset = datasets_dict[opts.dataset]
    
    # check the frames we need the dataloader to load
    frames_to_load = opts.frame_ids.copy()
    matching_ids = [0]
    if opts.use_future_frame:
        matching_ids.append(1)
    for idx in range(-1, -1 - opts.num_matching_frames, -1):
        matching_ids.append(idx)
        if idx not in frames_to_load:
            frames_to_load.append(idx)
    print('Loading frames: {}'.format(frames_to_load))
    if os.environ.get('DEBUG', '0') == '0':
        train_dataset = dataset(
            opts.data_path, train_filenames, opts.height, opts.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext)

        global_batch_size = opts.batch_size*get_num_gpus(gpus)
        train_loader = DataLoader(
            train_dataset, global_batch_size, True,
            num_workers=opts.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)

        val_dataset = dataset(
            opts.data_path, val_filenames, opts.height, opts.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext)

        val_loader = DataLoader(
            val_dataset, global_batch_size, False,
            num_workers=opts.num_workers, pin_memory=True, drop_last=True)
        lit_model = LitModel(opts, matching_ids)
        lit_trainer.fit(lit_model, train_loader, val_loader,
                        ckpt_path=opts.ckpt_path)

    else:
        gpus = 1
        opts.batch_size = 4
        global_batch_size = opts.batch_size*get_num_gpus(gpus)
        val_dataset = dataset(
            opts.data_path, val_filenames, opts.height, opts.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext)

        val_dataset.filenames = val_dataset.filenames[:100]
        item = val_dataset.__getitem__(0)
        val_loader = DataLoader(
            val_dataset, global_batch_size, False,
            num_workers=opts.num_workers, pin_memory=True, drop_last=True)

        lit_model = LitModel(opts, matching_ids)
        lit_trainer.fit(lit_model, val_loader, val_loader,
                        ckpt_path=opts.ckpt_path)