import torch

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path')
# args = parser.parse_args()

from manydepth.networks.depth_decoder import DepthDecoder
dd = DepthDecoder([ 64,  64, 128, 256, 512])
ckpt = torch.load('CityScapes_MR/old_depth.pth')

st = dd.state_dict()
new_ckpt = dict()
for k1, k2 in zip(ckpt.keys(), st.keys()):
    v1 = ckpt[k1]
    v2 = st[k2]
    print(v1.shape == v2.shape)
    print(k1, k2,v1.shape, v2.shape)
    new_ckpt[k2] = v1

torch.save(new_ckpt, 'CityScapes_MR/depth.pth')
# decoder.0.conv.conv.weight
# convs.upconv_4_0.conv.conv.weight