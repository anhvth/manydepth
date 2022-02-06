# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = nn.ModuleList(
            nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.Conv2d(256, 256, 3, 2, 1),

        )
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        # self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for conv_layer in range(self.convs):
            out = conv_layer(out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
