import logging 
from pathlib import Path 

from tqdm import tqdm 

import torch 

# REPRODUCIBILITY 
import random
import numpy as np 
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(5) # Default: 18 
# END REPRODUCIBILLITY 

def train(net, 
          device, 
          criterion, 
          optimizer, 
          lr_scheduler, 
          num_epochs, 
          dataloaders, 
          seed=18, 
          dirckp = 'checkpoints', 
          ckpname = 'checkpoint_epoch_best.pth', 
          threshold = 0.5, 
          n_classes = 2
          ): 
    '''
    IN: 
    dirckp: Path object, where to save the checkpoints. 

    OUT: 
    net: Module.nn object, classifier neural network in its best state.  
    '''
    

    # 0. Make training reproducible 
    set_seed(seed)
    
    # 1. Initialize logging 
    # Each epoch has a training phase and an optional validation phase 
    phases = ['train', 'val'] if 'val' in dataloaders.keys() else ['train']

    best_accuracy = 0 
    
    
    # 2. Begin epochs loop 
    for epoch in range(num_epochs):
        print('-' * 10)
        # Each epoch has a training phase and an optional validation phase 
        for phase in phases: 
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch}/{num_epochs-1}', unit='img') as pbar:
                if phase == 'train': 
                    net.train() # Set model to training mode 
                else: 
                    net.eval() # Set model to evaluate mode 
                
                running_loss = 0 
                running_accuracy = 0 
            
                # 3. Begin batches loop 
                for batch in dataloaders[phase]: 
                    inputs = batch['input'].to(device=device)
                    labels = batch['label'].to(device=device)

                    # Forward pass 
                    # Grad history is tracked IF in training mode 
                    with torch.set_grad_enabled(phase == 'train'):           
                        outputs = net(inputs) 
                        if n_classes > 2: 
                            probs = torch.softmax(outputs, dim=1)
                            preds = torch.argmax(outputs, dim=1) 
                            loss = criterion(probs, labels)
                        if n_classes == 2: 
                            probs = torch.sigmoid(outputs).flatten()
                            preds = torch.where(probs > threshold, 1, 0)
                            # Compute loss 
                            loss = criterion(probs.float(), labels.float())
                        
                        # Backward pass + Optimize IF in training phase 
                        if phase == 'train': 
                            optimizer.zero_grad() # zero the dloss/dx in each parameter 
                            loss.backward() # compute the new dloss/dx and add it to each parameter 
                            optimizer.step() # update the weights according to dloss/dx 
                        
                    # 4. Update running loss and acccuracy with batch loss and accuracy
                    running_loss += loss.item() * inputs.size(0)
                    running_accuracy += torch.sum(preds == labels.data)

                    # Update progress bar (tqdm) at the end of batch process 
                    pbar.update(inputs.size(0))
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                
            # 5. Update the learning rate scheduler at end of training epoch 
            if phase == 'train': 
                lr_scheduler.step(metrics=loss)
            
            # 6. Calculate the epoch loss and epoch accuracy 
            n_items = len(dataloaders[phase].dataset) 
            epoch_loss = running_loss / n_items 
            epoch_accuracy = running_accuracy / n_items 
            logging.info(f'{phase} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} LR: {lr_scheduler.get_last_lr()[0]:.4f}')

        # 7. Save best model checkpoint at the end of epoch 
        if epoch_accuracy > best_accuracy: 
            best_accuracy = epoch_accuracy 
            path_to_best_model_checkpoint = Path(dirckp) / ckpname
            torch.save(net.state_dict(), str(path_to_best_model_checkpoint))
            logging.info(f'Checkpoint {epoch} saved!')
    # Memory cleaning 
    if device == 'cuda':  
        del epoch_loss
        del epoch_accuracy
        del running_loss
        del running_accuracy
        del dataloaders
        torch.cuda.empty_cache()

    # Load saved model 
    print(str(path_to_best_model_checkpoint))
    net.load_state_dict(torch.load(str(path_to_best_model_checkpoint)))
    return net