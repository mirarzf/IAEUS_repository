# UI imports 
import tkinter as tk
from tkinter import filedialog, simpledialog

# File management imports 
from pathlib import Path 

# Data management imports 
import pandas as pd 

# Calculation imports 
import numpy as np 

# Image imports 
import cv2 as cv 
from PIL import Image, ImageTk

# CNN imports 
import torch 
from models.tvmodelsapi import load_from_tv
from customdataset import CustomDataset

# CAM imports 
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)
from cam_loader import computeCAM

# Segmentation visualization imports 
from segmentation_loader import convertCOCOJSONtoDF, create_poly

method = "gradcam++" # CAM method 

class ImageVisualizer:
    def __init__(self, root):
        # GET ALL IMAGES 
        img_folder = filedialog.askdirectory(title="Select folder containing images")
        self.img_list = [str(f) for f in Path(img_folder).iterdir() if f.is_file()]
        self.img_list.sort()        

        # GET COCO FILE FOR SEGMENTATION MASKS 
        self.coco_file = filedialog.askopenfilename(
            title="Select COCO JSON file containing segmentation masks", 
            filetypes=[(".json",".JSON")], 
            initialdir="D:/test_coco_json_segmentation_coord_changes/", 
            initialfile="dataset_coco_iaeus_i_segmentation_for_original.json")
        if self.coco_file == '': 
            self.segmentation_df = None 
        else: 
            self.segmentation_df = convertCOCOJSONtoDF(self.coco_file, img_folder)

        # SET CURRENT IMAGE 
        self.curr_index = 0
        self.original_img = cv.imread(self.img_list[0], cv.COLOR_RGB2BGR)
        self.visualized_img = cv.imread(self.img_list[0], cv.COLOR_RGB2BGR)
        self.width, self.height = self.original_img.shape[1], self.original_img.shape[0]

        # SET THE MODEL 
        # model_name = "resnet50"
        self.model_name = simpledialog.askstring(title="Choice of model type", prompt="Model Name", initialvalue="efficientnetv2")
        self.model_ckp = filedialog.askopenfilename(
            title="Select checkpoint of model to load", 
            filetypes=[("PTH", ".pth")], 
            initialdir="c:/Users/mirar/Endosono/checkpoints/", 
            initialfile=f"model_{self.model_name}.pth")
        self.n_classes = simpledialog.askinteger(title="The number of output classes", prompt="How many classes are there in the output?", initialvalue=2)
        self.categories = ["negative", "positive"] if self.n_classes == 2 else range(self.n_classes)

        if self.model_name != "": 
            model, _ = load_from_tv(self.model_name, n_classes=1 if self.n_classes == 2 else self.n_classes, freeze=False)
            loaded_state_dict = torch.load(self.model_ckp)
            model.load_state_dict(loaded_state_dict, strict=False)
            model.to(device=device)
            model.eval()
            self.model = model
        
        # SET CANVAS 
        self.root = root
        self.root.title("Segmentation and CAM visualizer")

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.canvas = tk.Canvas(self.image_frame)
        self.canvas.pack()
        
        # create the main sections of the layout, 
        # and lay them out
        bottom = tk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.prev_button = tk.Button(root, text="Previous", command=lambda: self.change_img(action="previous"))
        self.prev_button.pack(in_=bottom, side=tk.LEFT, padx=1)

        self.next_button = tk.Button(root, text="Next", command=lambda: self.change_img(action="next"))
        self.next_button.pack(in_=bottom, side=tk.RIGHT, padx=1)

        self.save_button = tk.Button(root, text="Save", command=self.save_img)
        self.save_button.pack(in_=bottom, side=tk.BOTTOM, padx=1)

        self.filename_label = tk.Label(root, text=f"Filename: {self.img_list[self.curr_index]}")
        self.filename_label.pack(in_=bottom, side=tk.TOP, padx=1)

        self.prediction_label = tk.Label(root, text="Prediction")
        if self.model_name != "": 
            self.prediction_label.pack(in_=bottom, side=tk.TOP, padx=1)

        self.cam_button = tk.Button(root, text="Show CAM", command=self.show_cam_button)
        self.cam_button.pack(in_=bottom, padx=1)
        if self.model_name == "": 
            self.cam_button.config(state="disabled")

        self.cam_mult_button = tk.Button(root, text="Show CAM with multiply", command=self.show_cam_mult_button)
        self.cam_mult_button.pack(in_=bottom, padx=1)
        if self.model_name == "": 
            self.cam_mult_button.config(state="disabled")
        
        self.seg_mask_button = tk.Button(root, text="Show segmentation mask", command=self.show_segmentation_mask)
        self.seg_mask_button.pack(in_=bottom, padx=1)
        if self.coco_file == "": 
            self.seg_mask_button.config(state="disabled")   
        
        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack(in_=bottom, side=tk.BOTTOM, padx=1)

        self.padx = 10 
        self.pady = 10
        self.canvas.config(width=self.width+1.5*self.padx, height=self.height+1.5*self.pady)
        
        # SET BOOLEANS FOR UI BUTTONS 
        self.showcam = False 
        self.showcammult = False 
        self.showsegmask = False 

        self.update()

    def change_img(self, action="next"): 
        # Button command 
        # Change global variables 
        if action == "next" and self.curr_index < len(self.img_list)-1: 
            self.curr_index += 1 
        elif action == "next" and self.curr_index == len(self.img_list)-1: 
            self.curr_index = 0 
        elif action == "previous" and self.curr_index > 0: 
            self.curr_index -= 1
        elif action == "previous" and self.curr_index == 0: 
            self.curr_index = len(self.img_list)-1
        self.filename_label.config(text=f"Filename: {self.img_list[self.curr_index]}")
        self.update()

    def save_img(self): 
        filesuffix = Path(self.img_list[self.curr_index]).suffix
        filename = Path(self.img_list[self.curr_index]).stem
        if self.showcam: 
            filename += "_cam"
        if self.showcammult: 
            filename += "_cammult"
        if self.showsegmask: 
            filename +="_mask"
        filename += filesuffix
        filename = tk.filedialog.asksaveasfilename(
            filetypes =[('PNG image','*.png'), ('JPG image','*.jpg, *.jpeg'), ('BITMAP image', '*.bmp')], 
            defaultextension=filesuffix, 
            initialfile=filename)
        cv.imwrite(filename, cv.cvtColor(self.visualized_img, cv.COLOR_RGB2BGR))

    def quit(self): 
        self.root.destroy()
    
    def show_cam_button(self): 
        # Button command 
        if self.showcam: 
            self.showcam = False 
            self.cam_button["text"] = "Show CAM"
        else: 
            self.showcam = True 
            self.cam_button["text"] = "Hide CAM"
        self.update()
    
    def show_cam_mult_button(self): 
        # Button command 
        if self.showcammult: 
            self.showcammult = False 
            self.cam_mult_button["text"] = "Show Multiply CAM"
        else: 
            self.showcammult = True 
            self.cam_mult_button["text"] = "Hide Multiply CAM"
        self.update()
        
    def show_segmentation_mask(self): 
        # Button command 
        if self.showsegmask: 
            self.showsegmask = False 
            self.seg_mask_button["text"] = "Show Segmentation Mask"
        else: 
            self.showsegmask = True 
            self.seg_mask_button["text"] = "Hide Segmentation Mask"
        self.update()

    def update(self): 
        self.original_img = cv.cvtColor(cv.imread(self.img_list[self.curr_index]), cv.COLOR_BGR2RGB)
        self.visualized_img = self.original_img
        self.photo = ImageTk.PhotoImage(image= Image.fromarray(cv.resize(self.visualized_img, dsize = (self.width, self.height))))
        
        batch, class_id = self.get_model_prediction()
        self.display_CAM(batch, class_id)
        self.display_CAM_and_multiply(batch, class_id)
        self.display_segmentation_mask()

        self.canvas.create_image(self.padx, self.pady, anchor=tk.NW, image=self.photo)
    
    def get_model_prediction(self): 
        if self.model_name != "": 
            # Begin model assessment 
            batch = CustomDataset.preprocess(Image.fromarray(self.original_img)).unsqueeze(0).to(device=device)
            if len(self.categories) == 2: 
                probs = self.model(batch).squeeze(0).sigmoid()
                score = probs.item()
                class_id = 1 if score > 0.5 else 0 
                if class_id == 1: 
                    self.canvas.config(bg='red')
                else: 
                    self.canvas.config(bg='#E4E4E4')
            else: 
                pred = self.model(batch).squeeze(0).softmax(0)
                class_id = pred.argmax().item()
                score = pred[class_id].item()
            category_name = self.categories[class_id]
            self.prediction_label.config(text=f"Prediction: {category_name}")
            # End model assessment 
            return batch, class_id 
        else: 
            return None, 0

    def display_CAM(self, batch, class_id): 
        if self.showcam and batch != None: 
            # Begin CAM heatmap computing 
            binaryclassification = len(self.categories) == 2
            gs_cam_img = computeCAM(
                net=self.model, 
                model_name=self.model_name, 
                img = self.original_img, 
                input_tensor=batch, 
                predicted_class=class_id, 
                cam_method=method, 
                binaryclassification=binaryclassification)
            cam_img = show_cam_on_image(self.original_img/255, gs_cam_img, use_rgb=True)
            # End CAM heatmap computing 
            self.visualized_img = cam_img
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(src = self.visualized_img, dsize = (self.width, self.height))))

    def display_CAM_and_multiply(self, batch, class_id): 
        if self.showcammult and batch != None: 
            # Begin CAM heatmap computing 
            binaryclassification = len(self.categories) == 2
            gs_cam_img = computeCAM(
                net=self.model, 
                model_name=self.model_name, 
                img = self.original_img, 
                input_tensor=batch, 
                predicted_class=class_id, 
                cam_method=method, 
                binaryclassification=binaryclassification)
            # End CAM heatmap computing 
            if len(self.original_img.shape) == 3: 
                gs_cam_img = np.repeat(gs_cam_img[:, :, np.newaxis], self.original_img.shape[2], axis=2)
            cam_multiplied = gs_cam_img * self.visualized_img
            self.visualized_img = np.uint8(cam_multiplied)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(src = self.visualized_img, dsize = (self.width, self.height))))
      
    def display_segmentation_mask(self, alpha=0.5): 
        if self.showsegmask: 
            filename = str(Path(self.img_list[self.curr_index]).name)
            seg_coords_rows = self.segmentation_df.loc[self.segmentation_df['file_name'] == filename]['segmentation']
            n = len(seg_coords_rows)
            mask = np.zeros(self.original_img.shape)
            for i in range(n): 
                seg_coords = seg_coords_rows.iloc[i]
                mask += create_poly(self.original_img.shape, seg_coords, color=(255,255,255))
            mask /= n
            imgWithSegMaskOverlay = np.uint(self.visualized_img+alpha*mask) 
            imgWithSegMaskOverlay = np.where(imgWithSegMaskOverlay > 255, 255, imgWithSegMaskOverlay)
            self.visualized_img = np.uint8(imgWithSegMaskOverlay)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.visualized_img).resize((self.width, self.height)))
          

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = tk.Tk()

    player = ImageVisualizer(root)
    root.mainloop()