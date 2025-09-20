# File reading imports 
from pathlib import Path 

# Calculations imports 
import numpy as np 

# UI imports 
import tkinter as tk
from tkinter import filedialog

# Image imports 
import cv2 as cv 

# CNN imports 
import torch 
from models.tvmodelsapi import load_from_tv
from customdataset import CustomDataset

# CAM imports 
from cam_loader import computeCAM

# Segmentation Mask Annotations imports 
from segmentation_loader import convertCOCOJSONtoDF, create_poly

#------------------------------------------------------------------------------------

# # Soft Segmentation DICE Score 
# Soft DICE 
def soft_DICE(cam, seg): 
    area_of_overlap = np.sum(seg*cam) 
    total_area = np.sum(cam)+np.sum(seg)
    if total_area > 0: 
        return 2.0*area_of_overlap/total_area 
    else: 
        return 1 if area_of_overlap == 0 else 0 

# 0. Get the images 
img_folder = filedialog.askdirectory(title="Select folder containing images")
coco_file = filedialog.askopenfilename(
            title="Select COCO JSON file containing segmentation masks", 
            filetypes=[(".json",".json")])
segmentation_df = convertCOCOJSONtoDF(coco_file, img_folder)
img_list = [str(f) for f in Path(img_folder).iterdir() if f.is_file() and (f.suffix == ".jpg" or f.suffix == ".png")]
nb_images = len(img_list)

# 1. Import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = tk.simpledialog.askstring(title="Choice of model type", prompt="Model Name", initialvalue="efficientnetv2")
# model_ckp = tk.simpledialog.askstring(title="Checkpoint to model to load", prompt="Path for model checkpoint", initialvalue=f"../Endosono/checkpoints/model_{model_name}_data_final_64batch.pth")
model_ckp = filedialog.askopenfilename(title="Checkpoint to model to load")
n_classes = tk.simpledialog.askinteger(title="The number of output classes", prompt="How many classes are there in the output?", initialvalue=2)
categories = ["negative", "positive"] if n_classes == 2 else range(n_classes)

if model_name != "": 
    model, _ = load_from_tv(model_name, n_classes=1 if n_classes == 2 else n_classes, freeze=False)
    loaded_state_dict = torch.load(model_ckp)
    model.load_state_dict(loaded_state_dict, strict=False)
    model.to(device=device)
    model = model
    model.eval()

# Set the CAM methods that will be chosen 
cam_method_names = ["gradcam", "scorecam", "gradcam++"]

# Set to 0 the metrics averaged on all images
average_DICES = [0 for i in range(len(cam_method_names))]
all_DICES = []

for filename in img_list: 

    # 2. Read the image 
    img = cv.imread(filename=filename)
    # cv.imshow(winname="Image", mat=img)
    # cv.waitKey(0)

    # 3. Get the prediction of original image 
    input_tensor = CustomDataset.preprocess(img=img).unsqueeze(0).to(device=device)
    original_score = model(input_tensor).squeeze(0).sigmoid().item()

    # 4. Get the CAM maps 
    multiple_cams_gs = []
    multiple_cams_rgb = []
    for cam_method in cam_method_names: 
        gs_cam_img = computeCAM(
                net=model, 
                model_name=model_name, 
                img = img, 
                input_tensor=input_tensor, 
                predicted_class=1, 
                cam_method=cam_method, 
                binaryclassification=(n_classes==2))
        multiple_cams_gs.append(gs_cam_img)
        if len(img.shape) == 3: 
            gs_cam_img = np.repeat(gs_cam_img[:, :, np.newaxis], img.shape[2], axis=2)
        multiple_cams_rgb.append(gs_cam_img)

    # 5. Get the masked image 
    multiple_masked_img = []
    for gs_cam_img in multiple_cams_rgb: 
        cam_masked_img = np.uint8(gs_cam_img*img) 
        multiple_masked_img.append(cam_masked_img)
        # cv.imshow(winname="CAM multiplied", mat=cam_masked_img)
        # cv.waitKey(0)

    # 6. Get the prediction of masked image 
    multiple_explanation_scores = []
    for cam_masked_img in multiple_masked_img: 
        masked_input_tensor = CustomDataset.preprocess(img=cam_masked_img).unsqueeze(0).to(device=device)
        explanation_score = model(masked_input_tensor).squeeze(0).sigmoid().item()
        multiple_explanation_scores.append(explanation_score)

    # 7. Get the original image segmentation mask 
    seg_coords_rows = segmentation_df.loc[segmentation_df['file_name'] == str(Path(filename).name)]['segmentation']
    if len(seg_coords_rows) > 0: 
        seg_coords = seg_coords_rows.iloc[0]
    else: 
        seg_coords = []
    mask = create_poly(img.shape, seg_coords)
    mask = mask[:,:,0]/255

    # 8. Create the segmentation masks based on CAM 
    thresh = 0.5
    multiple_cam_seg_masks = []
    for gs_cam_img in multiple_cams_rgb: 
        cam_mask = np.uint8(np.where(gs_cam_img > thresh, 1, 0))
        cam_mask = cam_mask[:,:,0]
        multiple_cam_seg_masks.append(cam_mask)

    # 7. Compute the metrics 
    # print("og score", original_score, "ex score", explanation_score)

    all_DICES_for_one_img = []
    for img_index, gs_cam_img in enumerate(multiple_cams_gs): 
        # print(np.min(gs_cam_img), np.min(mask))
        # print(np.max(gs_cam_img), np.max(mask))
        # cv.imshow("CAM GS",gs_cam_img)
        # cv.waitKey(0)
        # cv.imshow("SEGMENTATION MASK",255*mask)
        # cv.waitKey(0)
        average_DICES[img_index] += soft_DICE(gs_cam_img, mask)
        all_DICES_for_one_img.append(soft_DICE(gs_cam_img, mask))
        # print(soft_DICE(gs_cam_img, cam_mask))
    # print(average_DICES)
    all_DICES.append(all_DICES_for_one_img)

average_DICES = [e/nb_images for e in average_DICES]
print(average_DICES)

# import pandas as pd 

# df = pd.DataFrame(columns=("GradCAM", "ScoreCAM", "GradCAM++"))

# gradcam_col = [dices[0] for dices in all_DICES]
# scorecam_col = [dices[1] for dices in all_DICES]
# gradcampp_col = [dices[2] for dices in all_DICES]

# df['GradCAM'] = gradcam_col
# df['ScoreCAM'] = scorecam_col
# df['GradCAM++'] = gradcampp_col

# df.to_csv(f"./all_soft_dice_results_{model_name}.csv")

# root.mainloop()

# eps = 1e-8 # To avoid division by 0 error 
# average_drops = [0 for i in range(len(cam_method_names))]
# average_increase = [0 for i in range(len(cam_method_names))]
# for i, explanation_score in enumerate(multiple_explanation_scores): 

#     # Average Drop (from Grad-CAM++)
#     if original_score > 0: 
#         ad_i = max(0, original_score-explanation_score) / original_score 
#     else: 
#         ad_i = 0 
#     average_drops[i] += ad_i

#     # Average Increase = Increase in Confidence (from Grad-CAM++)
#     if explanation_score > original_score: 
#         average_increase[i] += 1 

# # Average Gain (from Opti-CAM)
# average_gain = np.argmax(average_drops)

# # Average Increase (from Opti-CAM)
# # average_increase = [ai/total_nb_img*100 for ai in average_increase]

# print(average_drops, average_increase, average_gain) 

# Insertion (from RISE)

# Deletion (from RISE)

# Proportion (from Score-CAM) - Localization ability 



### FAKE-CAM : https://arxiv.org/pdf/2104.10252