import argparse
import logging
from pathlib import Path

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from unet.unetutils.data_loading import MasterDataset
from unet.unetutils.dice_score import dice_loss
from evaluate import evaluate
from test import test_net
from unet.unet_model import UNet

from copy import deepcopy

from test import test_net

# REPRODUCIBILITY 
import random
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set as {seed}. \n")
# END REPRODUCIBILLITY 

# DATA DIRECTORIES 
## FOR TRAINING 
dir_img = Path('../data/train/pos')
dir_mask = Path('../data/segmentation/train/')

## FOR TESTING 
dir_img_test = Path('../data/test/pos/')
dir_mask_test = Path('../data/segmentation/test/')

## PARENT FOLDER OF CHECKPOINTS 
dir_checkpoint = Path('./checkpoints')

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = False,
              save_best_checkpoint: bool = True,
              amp: bool = False, 
              lossframesdecay: bool = False, 
              rgbtogs: bool = False, 
              foldnumber: int = 0):
    
    # 1. Choose data augmentation transforms (using albumentations) 
    geotransform = A.Compose([ 
        A.HorizontalFlip(p=0.5)
    ], 
    is_check_shapes=False, 
    additional_targets={'attmap':'mask', 'flox':'mask', 'floy':'mask'})
    colortransform = A.Compose([ 
        A.RandomBrightnessContrast(p=0.5), 
        A.ChannelShuffle(p=0.5)
    ])
    dataaugtransform = {'geometric': geotransform, 
                        'color': colortransform}
    dataaugtransform = dict() ################################################### COMMENT IF YOU WANT DATA AUGMENTATION 

    # 2. Split into train / validation partitions
    ids = [file.stem for file in dir_img.iterdir() if file.is_file() and str(file.name) != '.gitkeep']
    n_ids = len(ids)
    data_indices = list(range(n_ids))
    np.random.shuffle(data_indices)
    # Create folds if validation percentage is not 0 
    n_val = int(n_ids * val_percent)
    if n_val != 0: 
        k_number = n_ids // n_val
        last_idx_of_split = []
        q = n_ids // k_number 
        r = n_ids % k_number
        for i in range(k_number): 
            if i < r: 
                last_idx_of_split.append(i*q+1)
            else: 
                last_idx_of_split.append((i+1)*q)
        last_idx_of_split.append(n_ids)
        # Current fold number is (between [0;k-1]): 
        train_ids = [ids[idx] for idx in data_indices[:last_idx_of_split[foldnumber]]+data_indices[last_idx_of_split[foldnumber+1]:]] 
        val_ids = [ids[idx] for idx in data_indices[last_idx_of_split[foldnumber]:last_idx_of_split[foldnumber+1]]] 
    else: 
        train_ids = ids 
        val_ids = [] 
    n_train = len(train_ids)
    n_val = len(val_ids)
    val_percent = round(n_val/n_train,2) 

    # 3. Create datasets
    train_set = MasterDataset(images_dir=dir_img, masks_dir=dir_mask, file_ids=train_ids, transform=dataaugtransform, grayscale=rgbtogs) 
    val_set = MasterDataset(images_dir=dir_img, masks_dir=dir_mask, file_ids=val_ids, transform=dataaugtransform, grayscale=rgbtogs) 

    # 4. Create data loaders
    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')
    # For checkpoint saving 
    if save_checkpoint or save_best_checkpoint:
        adddirckp = 'U-Net-' + str(net.n_channels)
        if rgbtogs: 
            adddirckp += '-grayscale'
        dirckp = dir_checkpoint / adddirckp
        dirckp.mkdir(parents=True, exist_ok=True)

    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() 
    global_step = 0

    # 6. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                index = batch['index']


                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                index = index.to(device=device, dtype=torch.int)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    
                    if net.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    
                    if lossframesdecay: 
                        loss /= index 

                # optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round (validation testing at the end of epoch)
            net.eval()

            val_score = evaluate(net, val_loader, device)
            scheduler.step(val_score)
            net.train()

            logging.info('Validation Dice score: {}'.format(val_score))
            
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss) # Change learning rate 

        # 7. (Optional) Save checkpoint at each epoch 
        if save_checkpoint:            
            torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        
        # 8. (Optional) Save best model 
        if save_best_checkpoint: 
            if epoch == 1: 
                best_valscore = val_score 
                best_ckpt = 1 
                torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch_best.pth'))
                logging.info(f'Best checkpoint at {epoch} saved!')
                best_model_state = deepcopy(net.state_dict())
            else: 
                if val_score > best_valscore or n_val == 0: 
                    best_valscore = val_score
                    best_ckpt = epoch
                    torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch_best.pth'))
                    logging.info(f'Best checkpoint at {epoch} saved!')
                    best_model_state = deepcopy(net.state_dict())
            
            logging.info('Epoch loss: {}'.format(epoch_loss))
    
    return best_ckpt, best_model_state, val_ids

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--framesdecay', action='store_true', default=False, help='Modify loss function to add the frames lack of importance')
    parser.add_argument('--saveall', action='store_true', default=False, help='Save checkpoint at each epoch')
    parser.add_argument('--nosavebest', action='store_true', default=False, help="Don't save checkpoint of best epoch")
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    parser.add_argument('--test', action='store_true', default=False, help='Do the test after training')
    parser.add_argument('--viz', action='store_true', default=False, 
                        help='Visualize the images as they are processed')
    parser.add_argument('--foldnb', default=0, help='Number of the fold for cross-fold validation')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Setting seed for reproducibility 
    set_seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # if the model with attention is used, a different model will be loaded 
    n_channels = 3 
    if args.grayscale: 
        n_channels = 1 
    net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        modelToLoad = torch.load(args.load, map_location=device)
        net.load_state_dict(modelToLoad, strict=False)
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # TRAINING SECTION 
    try:
        best_ckpt, best_model_state, img_ids = train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            save_checkpoint=args.saveall,
            save_best_checkpoint=(not args.nosavebest),
            amp=args.amp, 
            lossframesdecay=args.framesdecay, 
            rgbtogs=args.grayscale, 
            foldnumber=args.foldnb)
        logging.info(f'Best model is found at checkpoint #{best_ckpt}.')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    # TESTING SECTION     
    if args.test: 
        logging.info(f'Testing dataset contains following ids : {img_ids}')
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        # if the model with attention is used, a different model will be loaded 
        testnet = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
        testnet.load_state_dict(best_model_state, strict=True) # Recover the best model state, the one we usually keep 
        testnet.to(device=device)
        logging.info(f'Start testing... ')
        test_DICE = test_net(
        testnet, 
        device=device,
        images_dir=dir_img_test, 
        masks_dir=dir_mask_test, 
        img_ids=[], 
        mask_threshold=0.5, 
        rgbtogs=args.grayscale, 
        savepred=False, 
        visualize=args.viz)
        logging.info(f'DICE score of testing dataset is: {test_DICE}')
        