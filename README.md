# IAEUS: official code for benchmarks of CNN for IAEUS dataset
This repository contains all the code for the benchmarks of the IAEUS dataset. 
There are three main folders: the code for the classification benchmarks, the code for the CAM visualizer and inference of the Soft Dice score between CAM and segmentation masks for images containing tumors and the code for segmentation. 

Data is expected to be put in the same architecture as the IAEUS dataset is introduced on https://iaeus.im-lis.com/. 
    |- train
    |   |- pos
    |   |- neg
    |- test
    |   |- pos
    |   |- neg
    |- segmentation 
    |   |- train 
    |   |- test 
    |   IA_EUS_TRAIN_COCO.json
    |   IA_EUS_TEST_COCO.json
