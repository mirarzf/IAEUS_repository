import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MasterDataset(Dataset): 
    def __init__(self, images_dir: str, masks_dir: str, file_ids: list, scale: float = 1.0, mask_suffix: str = '', transform = None, attmaps_dir: str = '', withatt: bool = True, flo_dir: str = '', withflo: bool = True, grayscale=False):
        self.withatt = withatt
        self.withflo = withflo

        self.grayscale = grayscale

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.attmaps_dir = Path(attmaps_dir)
        self.flo_dir = Path(flo_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        if transform != None: 
            if 'geometric' in transform.keys(): 
                self.geotransform = transform['geometric']
            else: 
                self.geotransform = None
            if 'color' in transform.keys(): 
                self.colortransform = transform['color']
            else: 
                self.colortransform = None 
        else: 
            self.geotransform = None 
            self.colortransform = None 

        self.ids = [] if len(file_ids) == 0 else file_ids
        # if not self.ids:
        #     raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} initial examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale=1, is_mask=False):
        w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # newW, newH = 572, 572
        newW, newH = 300, 300
        # assert newW > w or newH > h, 'Input images will be upsampled due to one dimension of the image being smaller than 572'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if is_mask: 
            # img_ndarray = np.where((img_ndarray == 1) | (img_ndarray == 2), 1, 0)[:,:,0] # Last index is to only keep one layer of image and not three channels for R, G and B.  
            # img_ndarray = np.where(img_ndarray > 0.5, 1, 0)[:,:,0]  
            # thres = np.quantile(img_ndarray, 0.75)
            thres = 0
            if img_ndarray.ndim == 3: 
                img_ndarray = np.where(img_ndarray > thres, 1, 0)[:,:,0] # Last index is to only keep one layer of image and not three channels with same value for (R,G,B). 
            else: 
                img_ndarray = np.where(img_ndarray > thres, 1, 0)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray
    
    @staticmethod
    def normalizeflow(flow_array): 
        flox = flow_array[:,:,0]
        floy = flow_array[:,:,1]
        doublenorms = np.sqrt(flox**2+floy**2)
        doublenorms = doublenorms[:,:,np.newaxis]
        doublenorms = np.concatenate((doublenorms, doublenorms), axis=2)
        return flow_array/doublenorms

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            # return Image.fromarray(np.load(filename))
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
    
    def getImageID(self, idx): 
        return self.ids[idx]

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        if self.withatt: 
            attmap_file = list(self.attmaps_dir.glob(name + '.*'))
        if self.withflo: 
            flo_file = list(self.flo_dir.glob(name + '.*'))
        
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}.'
        if self.withatt: 
            assert len(attmap_file) == 1, f'Either no attention map or multiple attention maps found for the ID {name}: {attmap_file}.'
        if self.withflo: 
            assert len(flo_file) == 1, f'Either no optical flow file or multiple attention optical flow files for the ID {name}: {flo_file}.'
        
        # Load the images 
        img = self.load(img_file[0])
        # If grayscale on, convert img 
        if self.grayscale: 
            img = img.convert('L')
        mask = self.load(mask_file[0])
        if self.withatt: 
            attmap = self.load(attmap_file[0])
        if self.withflo: 
            flo = self.load(flo_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        # Apply data augmentation 
        if self.geotransform != None: 
            if self.withatt and self.withflo: 
                transformed = self.geotransform(image=np.asarray(img), mask=np.asarray(mask), attmap=np.asarray(attmap), flox=flo[:,:,0], floy=flo[:,:,1])
                attmap = Image.fromarray(transformed['attmap'])
            elif self.withatt and not self.withflo: 
                transformed = self.geotransform(image=np.asarray(img), mask=np.asarray(mask), attmap=np.asarray(attmap))
                attmap = Image.fromarray(transformed['attmap'])
            elif not self.withatt and self.withflo: 
                transformed = self.geotransform(image=np.asarray(img), mask=np.asarray(mask), flox=flo[:,:,0], floy=flo[:,:,1])
            else: # No attention and no optical flow input 
                transformed = self.geotransform(image=np.asarray(img), mask=np.asarray(mask))
            if self.withflo: 
                flox = transformed['flox']
                floy = transformed['floy']
                flox = np.expand_dims(flox, axis=2)
                floy = np.expand_dims(floy, axis=2)
                flo = np.concatenate((flox, floy), axis=2)
            img = Image.fromarray(transformed['image'])
            mask = Image.fromarray(transformed['mask'])
        
        if self.colortransform != None: 
            img = Image.fromarray(self.colortransform(image=np.asarray(img))['image'])

        # Preprocess the images to turn them into an array 
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        if self.withatt: 
            attmap = self.preprocess(attmap, self.scale, is_mask=True)

        # Prepare getitem dictionary output with torch tensors 
        retdict = {}
        retdict['image'] = torch.as_tensor(img.copy()).float().contiguous()
        retdict['mask'] = torch.as_tensor(mask.copy()).long().contiguous()
        if self.withatt: 
            retdict['attmap'] = torch.as_tensor(attmap.copy()).long().contiguous()
        if self.withflo: 
            flo = self.normalizeflow(flo)
            flo = flo.transpose((2, 0, 1)) # Transpose dimensions of optical flow array so that they become (2, width, height) 
            flo_tensor = torch.as_tensor(flo.copy()).float().contiguous() # Transform into a tensor beforeapplying interpolation to change input size 
            # Then change the size of your tensor before adding it to the input dictionary 
            flo_tensor = flo_tensor.unsqueeze(0) 
            flo_tensor = torch.nn.functional.interpolate(input=flo_tensor, size=(300,300), mode='bicubic', align_corners=True)
            flo_tensor = flo_tensor.squeeze()
            # Add optical flow tensor to return dictionary 
            retdict['flow'] = flo_tensor
        
        retdict['index'] = idx+1
        return retdict

class BasicDataset(MasterDataset): 
    def __init__(self, images_dir: str, masks_dir: str, file_ids: list, scale: float = 1, mask_suffix: str = '', transform = dict(), grayscale=False):
        super().__init__(images_dir, masks_dir, file_ids, scale, mask_suffix, transform, attmaps_dir='', withatt=False, flo_dir='', withflo=False, grayscale=False)

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, file_ids: list, scale=1, transform = dict()):
        super().__init__(images_dir, masks_dir, file_ids, scale, mask_suffix='_mask', transform=transform)