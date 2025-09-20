# Image imports 
import cv2 as cv 

# CAM imports 
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# CAM METHODS DICTIONARY 
methods = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "gradcamelementwise": GradCAMElementWise, 
    # "opticam": Basic_OptCAM
}

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def computeCAM(net, 
               model_name, 
               img, 
               input_tensor, 
               predicted_class, 
               cam_method = "gradcam",
               binaryclassification = False): 
    # CAM TARGET LAYERS 
    if model_name[:6] == 'resnet': 
        target_layers = [net.layer4[-1]]
    elif model_name == "efficientnetv2": 
        target_layers = [net.features[-1]]
    elif model_name == "googlenet": 
        target_layers = [net.inception5b.branch4[-1].conv, net.inception5b.branch4[-1].bn]
    elif model_name == "vit": 
        target_layers = [net.encoder.layers.encoder_layer_11.ln_1]

    # CAM METHOD USED 
    cam_algorithm = methods[cam_method]

    # SPECIFIED CLASS TO COMPUTE CAM FOR  
    if binaryclassification: 
        targets = [ClassifierOutputTarget(0)] 
    else: 
        targets = [ClassifierOutputTarget(predicted_class)]

    # output = cv.cvtColor(img, cv.COLOR_BGR2RGB) # CAM fallback 

    # COMPUTE CAM 
    reshapeTransform = reshape_transform if model_name == "vit" else None
    with cam_algorithm(model=net, target_layers=target_layers, reshape_transform=reshapeTransform) as cam: 
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        # Get grayscale CAM 
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
        if binaryclassification: 
            grayscale_cam = grayscale_cam[0]
        else: 
            grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv.resize(grayscale_cam, dsize=(img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
    
    return grayscale_cam
