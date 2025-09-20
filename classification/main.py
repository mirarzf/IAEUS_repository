# LIBRARIES IMPORT
import argparse
import logging
from pathlib import Path 

from time import time 

from PIL import Image 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

# Import selfmade functions from other files 
from train import train, set_seed
from test import test
from models.tvmodelsapi import load_from_tv
from models.custommodel import CustomModel
from customloss import CustomLoss

# Import custom classes from other files 
from customdataset import CustomDataset

# DEFAULT DIRECTORIES OF ANNOTATED DATA 
train_folder = "../data/train"
test_folder = "../data/test"

# DEFAULT DIRECTORY FOR MODEL CHECKPOINTS 
dir_checkpoints = "./checkpoints"

# DEFAULT NAME OF MODEL TO LOAD 
model_to_load = "./checkpoint_epoch_best.pth"

# REPRODUCIBILITY 
set_seed(5) # Default : 18 
# END REPRODUCIBILITY 

# PARSER
def get_args():
    parser = argparse.ArgumentParser(description='Train, test or predict based on the neural network chosen')

    # Hyperparameters 
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--threshold', '-th', dest='threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--freeze', action='store_true', default=False, help='Only do transfer learning and freeze pretrained model weights')
    
    # Model specifications 
    parser.add_argument('--classes', '-c', dest='classes', type=int, default=2, help='Number of classes')
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model-name', '-m', type=str, dest='model_name', default='resnet50', help='Name of model class')
    model_group.add_argument('--custommodel', '-cm', dest='cm', action='store_true', default=False, help='Use custom model')
    parser.add_argument('--hub', action='store_true', default=False, help='Load pretrained torchvision model for pytorch hub')
    parser.add_argument('--load', '-f', type=str, default=False, help='Path to .pth file corresponding to weights to be loaded')

    # Dataset 
    parser.add_argument('--dir-train', type=str, dest='dir_train', default=train_folder, help='Get the folder containing the training data')
    parser.add_argument('--dir-val', type=str, dest='dir_val', default=False, help='Get the folder containing the validation data')
    parser.add_argument('--dir-test', type=str, dest='dir_test', default=test_folder, help='Get the folder containing the testing data')
    parser.add_argument('--dir-predict', type=str, dest='dir_predict', default=False, help='Get the folder containing the data on which to apply classification')
    parser.add_argument('--custom-tf', '-tf', action='store_true', dest='use_custom_tf', default=False, help='Use Custom Dataset transforms and not from \
                        default transformations from weights from torchhub')
    parser.add_argument('--noresize', action='store_false', dest="resize", default=True, help='Make false if you do not want the input data to be resized to (224,224) when preprocessed.')
    parser.add_argument('--augmentdata', '-da', action='store_true', dest='augment_data', default=False, help='Use online data augmentation')
    
    # Saving data 
    parser.add_argument('--ckp', type=str, dest='ckpfolder', default=dir_checkpoints, help='Path to the folder where to save \
                        checkpoints')
    parser.add_argument('--ckpname', type=str, dest='ckpname', default="checkpoint_epoch_best.pth", help='Name for checkpoint')

    # Main code behaviour 
    parser.add_argument('--train', action='store_true', default=False, help='Train mode')
    parser.add_argument('--test', action='store_true', default=False, help='Test mode')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict mode')

    # Other for training 
    parser.add_argument('--seed', '-s', type=int, default=18, dest='seed', help='Seed for reproducibility of the training')

    return parser.parse_args()

# HELPER FUNCTION TO LOG CONFUSION MATRIX PROPERLY 
def jolieConfMat(confmat, binary:bool = True, class_id:int = 0): 
    '''
    IN: 
    confmatrix: a dict representing confusion matrix for class class_id with keys 'tp', 'tn', 'fp', 'fn'. 

    OUT: 
    formatted_cm: a string to have a pretty visual of the confusion matrix in the logs 
    '''
    if binary: 
        cm_title = 'Confusion matrix \n'
    else: 
        cm_title = f'''Confusion matrix for class {class_id}\n'''
    pred_str = f'''{"":>21}{"Ground Truth"} \n{"":>21}{"Positive":>8} | {"Negative":>8} \n{"":>21}{"-"*19}\n'''
    pred_str += f"{'Prediction Positive':>19}" + " |{tp:>8} | {fp:>8} \n" + f"{'Negative':>19}" + " |{fn:>8} | {tn:>8} \n"

    formatted_cm = cm_title + pred_str.format(tp=confmat['tp'], fp=confmat['fp'], fn=confmat['fn'], tn=confmat['tn'])
    return formatted_cm

# MAIN CODE: TRAIN, TEST OR PREDICT

if __name__ == '__main__':
    # Get arguments from CLI
    args = get_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Use cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set up model
    # Number of output classes 
    n_classes = 1 if args.classes == 2 else args.classes 
    preprocess = None # Preprocessing to apply to the images in input of the model 
    
    # Create custom model 
    if args.cm: 
        model_name = "custom"
        net = CustomModel(out_classes=n_classes)
        preprocess = None 
        logging.info('Model is a custom one.')
    
    # Load pretrained models from torchvision 
    else: 
        net, preprocess = load_from_tv(modelname=args.model_name, n_classes=n_classes, freeze=args.freeze)
        model_name = args.model_name
        logging.info(f'Model loaded from torchvision hub')

    # Load weights from checkpoint 
    if args.load: 
        modelToLoad = torch.load(args.load, map_location=device)
        net.load_state_dict(modelToLoad, strict=False)
        logging.info(f'Model loaded weights from checkpoint: {args.load}')
    
    # Use CustomDataset preprocess function (and custom transforms defined in that preprocess function)
    if args.use_custom_tf: 
        logging.info('Preprocess used is the one defined in CustomDataset class. ')
        preprocess = None # use preprocess function defined in the CustomDataset class 
    
    net.to(device=device)
    logging.info(f'Network:\n'
                 f'Model name: {model_name}'
                 f'\t{n_classes} output channels (classes)')
    
    # TRAIN ## TO BE POLISHED 
    if args.train: 

        # Set the seed for reproducibility 
        set_seed(args.seed)
    
        # Load data # TO BE DONE  
        datasets = {} 
        
        # Add optional training dataset 
        if args.dir_train: 
            datasets['train'] = CustomDataset(datafolder=args.dir_train, preprocess_fct=preprocess, dataaug=args.augment_data, resize=args.resize)
        # Add optional validation dataset 
        if args.dir_val: 
            datasets['val'] = CustomDataset(datafolder=args.dir_val, preprocess_fct=preprocess, resize=args.resize)
        
        # Create dataloaders 
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                    for x in datasets.keys()} 
        # dataloaders is a dictionary of DataLoader with *at most* two keys: 'train' and 'val'. 

        # Hyperparameters: 
        if n_classes > 2: 
            criterion = nn.CrossEntropyLoss()
        else: 
            # criterion = nn.BCEWithLogitsLoss()
            criterion = CustomLoss()
        
        optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-3) 
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)  
        
        logging.info(f'''Starting training:
            Epochs:           {args.epochs}
            Batch size:       {args.batch_size}
            Learning rate:    {args.lr}
            Training size:    {len(datasets["train"])}
            Validation size:  {0 if not args.dir_val else len(dataloaders['val'])}
            Checkpoints path: {args.ckpfolder + "/" + args.ckpname}
            Device:           {device.type}
        ''')    
        
        t_begin_train = time()
        net = train(
            net=net, 
            device=device,
            criterion=criterion, 
            optimizer=optimizer, 
            lr_scheduler=scheduler, 
            num_epochs=args.epochs, 
            dataloaders=dataloaders, 
            seed=args.seed, 
            modelname=args.model_name, 
            dirckp=args.ckpfolder, 
            ckpname=args.ckpname, 
            threshold=args.threshold,
            n_classes=args.classes
            )
        total_train_time = time() - t_begin_train
        
        logging.info(f'''Training ended.
                     Total time for training: {total_train_time:.2f}
                     Best model saved at {args.ckpfolder + "/" + args.ckpname}. 
        ''')

    # TEST 
    if args.test: 

        # Add optional testing dataset 
        test_dataset = CustomDataset(datafolder=args.dir_test, preprocess_fct=preprocess, resize=args.resize)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
        
        logging.info(f'''Starting testing: 
            Testing size:   {len(test_dataset)}
            Device:         {device.type}
        ''')

        # NOTE: If --train and --test are given to the CLI, the tested network is the trained network 
        t_begin_test = time()
        scores = test(
            net=net, 
            device=device, 
            dataloader=test_dataloader, 
            n_classes=args.classes, 
            threshold=args.threshold, 
            binaryclass=(args.classes==2)
        )
        total_test_time = time() - t_begin_test
        
        logging.info(f'''Testing ended.
                     Total time for testing: {total_test_time:.2f} 
        ''')

        # Separate multiclass logging and logging for binary classification
        multiscores_keys = []
        uniquescore_key = []
        for key in scores.keys(): 
            if type(scores[key]) == list and len(scores[key]) > 1 and key != 'confusion_matrices': 
                multiscores_keys.append(key) 
            elif key != 'confusion_matrices': 
                uniquescore_key.append(key)

        # Logging for multiclass 
        if args.classes > 2: 
            align_left_mc = max(map(len, multiscores_keys))+1 # used for alignment of results in logging 
            for i in range(args.classes): 
                logging.info(f'''Scores for class {i}''')
                logging.info(jolieConfMat(scores['confusion_matrices'][i], binary = False, class_id = i))
                for key in multiscores_keys: 
                    logging.info(f'''{key:<{align_left_mc}}: {scores[key][i]:.4f}''')
                print('-'*10)
        
        # Logging for unique scores 
        align_left_unique = max(map(len, uniquescore_key))+1 # used for alignment of results in logging 
        if args.classes == 2: 
            logging.info(jolieConfMat(scores['confusion_matrices'][0]))
        for key in uniquescore_key: 
            if type(scores[key]) == list and len(scores[key]) == 1: 
                score = scores[key][0]
            elif type(scores[key]) != list: 
                score = scores[key]
            logging.info(f'''{key:<{align_left_unique}}: {score:.4f}''')
                    
    # PREDICT 
    if args.predict: 
        # 1. Put model in eval mode and choose categories 
        net.eval()
        if args.classes == 2: 
            categories = ["negative", "positive"]
        else: 
            categories = [str(i) for i in range(args.classes)]

        # 2. Prep data to be loaded for prediction 
        imgs_to_predict = [f for f in Path(args.dir_predict).iterdir() if f.is_file() and (f.suffix == ".jpg" or f.suffix == ".png")]

        # 3. Start the loop of predictions  
        for img_path in imgs_to_predict: 
            img = Image.open(str(img_path))

            # 4. Apply inference preprocessing transforms 
            if preprocess == None or args.use_custom_tf: 
                batch = CustomDataset.preprocess(img)
            else: 
                batch = preprocess(img)
            batch = batch.unsqueeze(0).to(device=device)

            #5. Inference 
            if args.classes == 2: 
                probs = net(batch).squeeze(0).sigmoid()
                score = probs.item()
                class_id = 1 if score > args.threshold else 0 
            else: 
                pred = net(batch).squeeze(0).softmax(0)
                class_id = pred.argmax().item()
                score = pred[class_id].item()
            category_name = categories[class_id]
            # logging.info(f"Prediction for {img_path.name} is {category_name}: {100 * score:.2f}%")
