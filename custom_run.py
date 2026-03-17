import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
from utils.frame_utils import writeFlow

from xv_file import scan_files, get_file_name

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(outpath_flo, outpath_vis, outpath_heat, filename, args, model, image1, image2):
    #os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    flow, info = calc_flow(args, model, image1, image2)
    flow_cpu = flow[0].permute(1, 2, 0).cpu().numpy()
    writeFlow(os.path.join(outpath_flo, f'{filename}.flo'), flow_cpu)   #(f"{path}flow_{i:06d}.flo", flow_cpu)
    flow_vis = flow_to_image(flow_cpu, convert_to_bgr=True)
    cv2.imwrite(os.path.join(outpath_vis, f'{filename}.jpg'), flow_vis) #cv2.imwrite(f"{path}flow.jpg", flow_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(os.path.join(outpath_heat, f'{filename}.jpg'), image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def demo_custom(model, args, filename1, filename2, outpath_flo, outpath_vis, outpath_heat, device=torch.device('cuda')):
    image1 = cv2.imread(filename1) # "./custom/image1.jpg"
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(filename2) # "./custom/image2.jpg"
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = cv2.resize(image1, None, fx=0.5, fy=0.5)
    image2 = cv2.resize(image2, None, fx=0.5, fy=0.5)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    demo_data(outpath_flo, outpath_vis, outpath_heat, get_file_name(filename1)[1], args, model, image1, image2) #'./custom/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser)
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    #& C:/Users/jdibe/miniconda3/envs/SEA-RAFT/python.exe d:/jcds/Documents/GitHub/SEA-RAFT/custom_run.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth

    files_l = sorted(scan_files('C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\X_etna_l'))
    files_r = sorted(scan_files('C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\X_etna_r'))

    files_f1 = zip(files_l, files_l[1:])
    files_f2 = zip(files_l, files_l[2:])
    files_d0 = zip(files_l, files_r)

    for filename1, filename2 in files_f1:
        print(f'{filename1} -> {filename2}')
        demo_custom(model, args, filename1, filename2, "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\flow", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\vis_flow", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\heat_flow", device=device)
        #break

    for filename1, filename2 in files_f2:
        print(f'{filename1} -> {filename2}')
        demo_custom(model, args, filename1, filename2, "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\flow2", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\vis_flow2", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\heat_flow2", device=device)
        #break

    for filename1, filename2 in files_d0:
        print(f'{filename1} -> {filename2}')
        demo_custom(model, args, filename1, filename2, "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\disp", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\vis_disp", "C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\heat_disp", device=device)
        #break

if __name__ == '__main__':
    main()
