import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet.unetutils.data_loading import MasterDataset
from unet.unet_model import UNet, UNetAtt
from unet.unetutils.utils import plot_img_and_mask
from test import mask_to_image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# CHOOSE INPUT DIRECTORIES 
## RGB input 
imgdir = Path("./data/segmentation/test")
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 

## Folder where to save the predicted segmentation masks 
outdir = Path("./results/unet")

## Checkpoint directories 
dir_checkpoint = Path('./checkpoints')
### Model file path inside dir_checkpoint folder 
ckp = "U-Net-3/checkpoint_epoch_best.pth"


def predict_img(net,
                full_img: Image,
                out_filename: str, 
                device,
                mask_threshold: float =0.5, 
                rgbtogs: bool = False, 
                savepred: bool = True, 
                visualize: bool = False):
    net.eval()
    imgsize = full_img.size
    # If grayscale on, convert img 
    if rgbtogs: 
        full_img = full_img.convert('L')
    img = torch.from_numpy(MasterDataset.preprocess(full_img, is_mask=False))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # predict the mask 
        mask_pred = net(img)
            
        if net.n_classes == 1:
            # convert to one-hot format
            mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()
        else:
            mask_pred = mask_pred.argmax(dim=1)
        
        mask_pred_img = mask_to_image(mask_pred[0].cpu().numpy(), net.n_classes, imgsize)
        if savepred: 
            logging.info(f"Mask saved to {out_filename}")
            mask_pred_img.save(out_filename)
        if visualize:
            plot_img_and_mask(img[0].cpu().numpy().transpose((1,2,0))[:,:,:3], mask_pred[0].cpu().numpy(), net.n_classes) 
            logging.info(f'Visualizing results for image {filename}, close to continue...')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    inputgroup.add_argument('--dir', action='store_true', default=False, help='Use directories specified in predict.py file instead')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    parser.add_argument('--noimg', action='store_true', default=False, help='No image as input')

    return parser.parse_args()


def get_output_filenames(args): 
    return args.output or [outdir / f.name for f in imgfilenames]
    

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Prepare the input files 
    if args.dir: 
        in_files = imgfilenames
    else: 
        in_files = args.input
    out_files = get_output_filenames(args)

    n_channels = 3 
    if args.grayscale: 
        n_channels = 1 
    net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    modelToLoad = torch.load(args.model, map_location=device)
    nchanToLoad = modelToLoad['inc.double_conv.0.weight'].shape[1]
    assert net.n_channels == nchanToLoad, \
        f"Number of input channels ({net.n_channels}) and loaded model ({nchanToLoad}) are not the same. Choose a different model to load."
    net.load_state_dict(modelToLoad, strict=False)
    net.to(device=device)

    logging.info('Model loaded!')

    for i, filename in enumerate(tqdm(in_files)):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        
        out_filename = out_files[i]
        predict_img(net=net,
                    full_img=img,
                    out_filename=out_filename, 
                    device=device, 
                    mask_threshold=args.mask_threshold,
                    rgbtogs=args.grayscale, 
                    savepred=not args.no_save, 
                    visualize=args.viz)