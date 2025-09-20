import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader
from unet.unetutils.data_loading import MasterDataset, BasicDataset
from unet.unetutils.dice_score import multiclass_dice_coeff, dice_coeff
from unet.unet_model import UNet
from unet.unetutils.utils import plot_img_and_mask_and_gt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # REPRODUCIBILITY 
# import random
# def set_seed(seed: int = 42) -> None:
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # When running on the CuDNN backend, two further options must be set
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     logging.info(f"Random seed set as {seed}. \n")
# # END REPRODUCIBILLITY 

# CHOOSE INPUT DIRECTORIES 
## RGB input 
imgdir = Path("./data/test/pos")
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 

## Ground truth masks 
gtdir = Path("./data/segmentation/test")
# gtdir = Path("./data/test/masks-randomsplit")

## Folder where to save the predicted segmentation masks 
outdir = Path("./results/")
# outdir = None 

## Checkpoint directories 
dir_checkpoint = Path('./checkpoints')
### Model file path inside dir_checkpoint folder 
ckp = "U-Net-3/checkpoint_epoch_best.pth"

##########################################################################################

def mask_to_image(mask: np.ndarray, n_classes = 2, finalsize = (0,0)):
    img = Image.fromarray((mask * 255 / (n_classes-1)).astype(np.uint8))
    if finalsize == (0,0): 
        finalsize = img.size
    return img.resize(finalsize)

def test_net(net, 
              device,
              images_dir, 
              masks_dir, 
              img_ids: list = [], 
              img_scale: float = 1.0,
              mask_threshold: float = 0.5, 
              rgbtogs: bool = False, 
              savepred: bool = False, 
              visualize: bool = False):
    # 1. Create dataset
    if len(img_ids) == 0: 
        ids = [file.stem for file in images_dir.iterdir() if file.is_file() and file.name != '.gitkeep']
    else: 
        ids = img_ids 
    test_set = MasterDataset(images_dir=images_dir, masks_dir=masks_dir, file_ids=ids, scale=img_scale, grayscale=rgbtogs) 

    # 2. Create data loader 
    loader_args = dict(num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, **loader_args)
    
    logging.info(f'''Starting testing:
        Attention model: {useatt}
        Testing size:    {len(test_set)}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 3. Calculate DICE score and save predicted masks if toggled on 
    net.eval()

    num_val_batches = len(test_loader)
    dice_score = 0
    
    # In the case of no rgb input 
    lastimgchannel = 3
    if rgbtogs: 
        lastimgchannel = 1

    # iterate over the test set
    for batch in tqdm(test_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        one_hot_mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        
        # predict and compute DICE score 
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                # convert to one-hot format
                mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, one_hot_mask_true, reduce_batch_first=False)
            else:
                mask_pred = mask_pred.argmax(dim=1)
                # convert to one-hot format
                one_hot_mask_pred = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background (because multiclass)
                dice_score += multiclass_dice_coeff(one_hot_mask_pred[:, 1:, ...], one_hot_mask_true[:, 1:, ...], reduce_batch_first=False)

            # 4. Save or visualize predicted masks if toggled on 
            if savepred or visualize: 
                index = batch['index']-1
                filename = test_set.getImageID(index) + '.png'
                imgsize = Image.open(imgdir / filename).size
                if savepred: 
                    logging.info(f"Saving prediction of {filename}")
                    mask_pred_img = mask_to_image(mask_pred[0].cpu().numpy(), net.n_classes, imgsize)
                    mask_pred_img.save(outdir / filename)
                if visualize:
                    plot_img_and_mask_and_gt(image[0].cpu().numpy().transpose((1,2,0))[:,:,:3], mask_true[0].cpu().numpy(), mask_pred[0].cpu().numpy(), net.n_classes) 

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


##########################################################################################

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', default=imgdir, help='Directory of input images')
    parser.add_argument('--ground_truth', '-gt', metavar='GROUND TRUTH', default=gtdir, help='Directory of ground truth masks')
    parser.add_argument('--output', '-o', metavar='OUTPUT', default=outdir, help='Directory for output images')
    parser.add_argument('--viz', '-v', action='store_true', default=False, 
                        help='Visualize the images as they are processed')
    parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # set_seed(0)

    useatt = True if args.input_att != None else False 
    useflow = True if (args.flow and args.input_flow != None) else False 

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
    net.load_state_dict(modelToLoad, strict=True)
    net.to(device=device)

    logging.info('Model loaded!')

    savepred = True if outdir != None else False 
    dice_score = test_net(
        net, 
        device=device,
        images_dir=imgdir, 
        masks_dir=gtdir, 
        img_scale=args.scale,
        mask_threshold=args.mask_threshold, 
        rgbtogs=args.grayscale, 
        savepred=savepred, 
        visualize=args.viz)
        
    logging.info(f'Final average DICE score is: {dice_score}')