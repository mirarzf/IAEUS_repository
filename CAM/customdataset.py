from pathlib import Path 

from PIL import Image
import numpy as np 
import albumentations as A

import torch 
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CustomDataset(Dataset): 
    def __init__(self, 
                 datafolder:str, 
                 preprocess_fct=None, 
                 dataaug=False, 
                 ) -> None:
        
        # Attributes initialization 
        self.images = [] # List of all images 
        self.labels = [] # List of all labels. labels[i] is the label of images[i]
        classes_folder = [dir for dir in Path(datafolder).iterdir() if dir.is_dir()]
        classes_folder.sort()
        self.label_names = [str(dir.stem) for dir in classes_folder]
        self.dataaug = dataaug 

        # Complete self.images 
        for class_idx, class_folder in enumerate(classes_folder): 
            all_images_in_class_idx = [f for f in class_folder.iterdir() if f.is_file()]
        ######################################## EXPERIMENT TO HAVE A VERY UNBALANCED DATASET 
            # if class_folder.stem == "pos": 
            #     all_images_in_class_idx = all_images_in_class_idx[:70]
            # print(len(all_images_in_class_idx))
            self.images += all_images_in_class_idx
            self.labels += [class_idx for i in range(len(all_images_in_class_idx))]

        # If Custom Dataset uses the preprocess of a usual model 
        if preprocess_fct != None: 
            self.preprocess_fct = preprocess_fct
        else: 
            self.preprocess_fct = None 

    def __len__(self):
        return len(self.images)
    
    @staticmethod # can make preprocess a static method so it is used outside of the class 
    def preprocess(img, label = 0, dataaug = False): 
        ## TO BE POLISHED UPON CREATION OF CUSTOM CNN 
        preprocessed_img = img 
        
        # Parameters for transforms to apply 
        resize_dim = [224,224]

        if not dataaug: 
            # transforms = v2.Compose([
            #     v2.ToImage(), 
            #     v2.Resize(resize_dim, interpolation=v2.InterpolationMode.BILINEAR), 
            #     v2.ToDtype(torch.float32, scale=True)
            # ])
        
            # preprocessed_img = transforms(preprocessed_img)
            # return preprocessed_img 
            
            transforms = A.Compose([
                A.Resize(width=resize_dim[0], height=resize_dim[1], p=1)
            ])
            preprocessed_img = np.array(preprocessed_img)
            preprocessed_img = transforms(image=preprocessed_img)["image"]
            preprocessed_img = torch.from_numpy(np.array([preprocessed_img[:,:,0], preprocessed_img[:,:,1], preprocessed_img[:,:,2]])) # .type(torch.float32)
            tofloat = v2.ToDtype(torch.float32, scale=True)
            preprocessed_img = tofloat(preprocessed_img)

            return preprocessed_img 
        else: 
            ptransform = 0.5 if label == 1 else 0.2
            transforms = A.Compose([
                A.Resize(width=resize_dim[0], height=resize_dim[1], p=1),
                A.ShiftScaleRotate(
                    shift_limit = 1, #0
                    scale_limit=1, #(0,1), 
                    rotate_limit=1, #0
                    p=ptransform),
                A.HorizontalFlip(p=ptransform),
                # A.RandomBrightnessContrast(
                #     brightness_limit=0.5, #(0,0.2), 
                #     contrast_limit=0.5, #0, 
                #     p=ptransform),
            ])
            preprocessed_img = np.array(preprocessed_img)
            augmented_img = transforms(image=preprocessed_img)["image"]
            augmented_img = torch.from_numpy(np.array([augmented_img[:,:,0], augmented_img[:,:,1], augmented_img[:,:,2]])) # .type(torch.float32)
            tofloat = v2.ToDtype(torch.float32, scale=True)
            augmented_img = tofloat(augmented_img)
            return augmented_img 
    
    def __getitem__(self, index):
        # Initialize the dictionary to return input and label at index 
        retdict = {}

        # Get the image and label 
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]

        # Preprocess the image to be returned as a tensor 
        if self.preprocess_fct != None: 
            processed_image = self.preprocess_fct(image)
        else: 
            processed_image = self.preprocess(image, label, self.dataaug) 
        
        # Return the dictionary 
        retdict['input'] = processed_image 
        retdict['label'] = label # Value is 1 or 0 depending on label 
        return retdict 