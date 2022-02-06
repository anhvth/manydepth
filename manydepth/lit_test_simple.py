# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from manydepth import networks
from manydepth.options import MonodepthOptions
from .layers import transformation_from_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--target_image_path', type=str,
                        help='path to a test image to predict for', required=True)
    parser.add_argument('--source_image_path', type=str,
                        help='path to a previous image in the video sequence', required=True)
    parser.add_argument('--intrinsics_json_path', type=str,
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.eye(4)
    with open(intrinsics_path, 'r') as f:
        K[:3, :3] = np.array(json.load(f))

    # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
    # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    # Loading pretrained model
    # print("   Loading pretrained encoder")
    options = MonodepthOptions()
    opts = options.parser.parse_known_args()[0]
    opts.height = 192
    opts.width = 512
    # import ipdb; ipdb.set_trace()
    from manydepth.lit_train import prepare_model
    models = prepare_model(opts, 2)
    ckpt = torch.load(
        'lightning_logs/cityscape/Jan18-03:13:14/ckpts/last.ckpt')
    state_dict = dict()
    for k, v in ckpt['state_dict'].items():
        if k.startswith('models.'):
            state_dict[k.split('models.')[1]] = v

    ks = [k for k in state_dict.keys() if k.startswith('pose.net')]
    for k in ks:
        del state_dict[k]
    results = models.load_state_dict(state_dict)
    print(results)
    encoder = models['encoder']
    depth_decoder = models['depth']
    pose_enc = models['pose_encoder']
    pose_dec = models['pose']

    from manydepth.lit_train import LitModel

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    pose_enc.eval()
    pose_dec.eval()

    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    # Load input data
    input_image, original_size = load_and_preprocess_image(args.target_image_path,
                                                           resize_width=opts.width,
                                                           resize_height=opts.height)

    source_image, _ = load_and_preprocess_image(args.source_image_path,
                                                resize_width=opts.width,
                                                resize_height=opts.height)
    K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                             resize_width=opts.width,
                                             resize_height=opts.height)

    with torch.no_grad():

        # Estimate poses
        pose_inputs = [source_image, input_image]
        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = pose_dec(pose_inputs)
        pose = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)

        if args.mode == 'mono':
            pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
            source_image *= 0

        # Estimate depth
        # import ipdb; ipdb.set_trace()
        output, lowest_cost, _ = encoder(current_image=input_image,
                                         lookup_images=source_image.unsqueeze(
                                             1),
                                         poses=pose.unsqueeze(1),
                                         K=K,
                                         invK=invK,
                                         min_depth_bin=opts.min_depth,
                                         max_depth_bin=opts.max_depth)

        output = depth_decoder(output)

        sigmoid_output = output[("disp", 0)]
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False)
        sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

        # Saving numpy file
        directory, filename = os.path.split(args.target_image_path)
        output_name = os.path.splitext(filename)[0]
        name_dest_npy = os.path.join(
            directory, "{}_disp_{}.npy".format(output_name, args.mode))
        np.save(name_dest_npy, sigmoid_output.cpu().numpy())

        # Saving colormapped depth image and cost volume argmin
        for plot_name, toplot in (('costvol_min', lowest_cost), ('disp', sigmoid_output_resized)):
            toplot = toplot.squeeze()
            normalizer = mpl.colors.Normalize(
                vmin=toplot.min(), vmax=np.percentile(toplot, 95))
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(toplot)[
                              :, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(directory,
                                        "{}_{}_{}.jpeg".format(output_name, plot_name, args.mode))
            im.save(name_dest_im)

            print("-> Saved output image to {}".format(name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
