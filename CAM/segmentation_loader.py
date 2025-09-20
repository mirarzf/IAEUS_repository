from pathlib import Path 

import pandas as pd 
import json 

import numpy as np
import cv2 as cv 

def create_poly(img_shape, seg_coords, color=(255,0,0)): 
    '''
    IN: 
    img_shape: tuple (w, h, 3) with width and height of image 
    seg_coords: list of tuples of int (x, y) of coordinates for the summits of the \
        polygon of the mask 
    color: a (B,G,R) tuple of the intended color for the segmented portion  

    OUT: 
    mask: a numpy array with 0 for the background and color for the segmented part 
    '''
    mask = np.zeros(img_shape)
    if len(seg_coords) > 0: 
        seg_coords_array = np.array(seg_coords, dtype=int)
        cv.fillPoly(mask, pts=[seg_coords_array], color=color)
    return mask 
    
def reformatSegmentationRow(seg_part): 
    # Avoid nested lists 
    if type(seg_part) is list: 
        segmentation = []
        for e in seg_part: 
            if type(e) is list: 
                segmentation += e 
            else: 
                segmentation.append(e)
        seg_coords = [(segmentation[i], segmentation[i+1]) for i in range(0,len(segmentation), 2)]
    else: 
        seg_coords = []
    return seg_coords

def convertCOCOJSONtoDF(viaproject, img_dir): 
    '''
    IN: 
    viaproject: the path to the COCO JSON file exported from VIA annotator 
    img_dir: the directory containing the images from the COCO JSON file 

    OUT: 
    ret_df: a pandas DataFrame with columns being \
        ['id', 'file_name', 'width', 'height', 'segmentation']
    '''
    # Load JSON file 
    f = open(viaproject)
    dico = json.load(f)
    f.close()

    # Join images and annotations information in the same DataFrame 
    images_df = pd.DataFrame.from_dict(dico['images'])
    annotations_df = pd.DataFrame.from_dict(dico['annotations'])
    images_df = images_df[['id', 'width', 'height', 'file_name']]
    annotations_df = annotations_df[['id', 'image_id', 'segmentation', 'area', 'bbox']]
    ret_df_list = []
    for ind in images_df.index: 
        # Change width and height 
        imgpath = Path(img_dir) / str(images_df.iloc[ind]['file_name'])
        if imgpath.exists(): 
            image = cv.imread(str(imgpath))
            w, h = image.shape[0], image.shape[1]
            images_df.at[ind, 'width'] = w 
            images_df.at[ind, 'height'] = h 
            # Merge with annotations Dataframe 
            annot_rows_df = annotations_df.loc[annotations_df['image_id'] == images_df.iloc[ind]['id']]
            # Reformat segmentation coordinates list 
            for i in annot_rows_df.index: 
                annot_rows_df.at[i, 'segmentation'] = reformatSegmentationRow(annot_rows_df.at[i, 'segmentation'])
            if len(annot_rows_df) > 0: 
                new_rows = pd.merge(left = images_df.iloc[[ind]], right = annot_rows_df, how = 'cross')
                ret_df_list.append(new_rows)
    ret_df = pd.concat(ret_df_list, ignore_index=True)

    # segmentations = ret_df[['segmentation']]
    columns_to_keep = ['file_name', 'width', 'height', 'segmentation']
    ret_df = ret_df[columns_to_keep]

    return ret_df 