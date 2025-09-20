import torch.nn as nn 
import torchvision.models 
from torchvision.models import ( 
    ResNet18_Weights,  
    ResNet34_Weights,  
    ResNet50_Weights,  
    ResNet101_Weights,  
    ResNet152_Weights,   
    EfficientNet_V2_M_Weights, 
    GoogLeNet_Weights, 
    ViT_B_16_Weights
)

# Load pretrained ResNet on ImageNet 
def load_resnet_from_tv(modelname, n_classes, freeze = True): 
    if modelname == 'resnet18': 
        weights = ResNet18_Weights.DEFAULT
        model = torchvision.models.get_model("resnet18", weights="DEFAULT")
    elif modelname == 'resnet34': 
        weights = ResNet34_Weights.DEFAULT
        model = torchvision.models.get_model("resnet34", weights="DEFAULT")
    elif modelname == 'resnet50': 
        weights = ResNet50_Weights.DEFAULT
        model = torchvision.models.get_model("resnet50", weights="DEFAULT")
    elif modelname == 'resnet101': 
        weights = ResNet101_Weights.DEFAULT
        model = torchvision.models.get_model("resnet101", weights="DEFAULT")
    elif modelname == 'resnet152': 
        weights = ResNet152_Weights.DEFAULT
        model = torchvision.models.get_model("resnet152", weights="DEFAULT")
    
    if freeze: 
        for param in model.parameters(): 
            param.requires_grad = False 
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    for param in model.fc.parameters(): 
        param.requires_grad = True 
    
    return model, weights.DEFAULT.transforms()

# Load pretrained GoogLeNet on ImageNet 
def load_googlenet_from_tv(n_classes, freeze = True): 
    weights = GoogLeNet_Weights.DEFAULT
    model = torchvision.models.get_model("googlenet", weights="DEFAULT")
    
    if freeze: 
        for param in model.parameters(): 
            param.requires_grad = False 
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    for param in model.fc.parameters(): 
        param.requires_grad = True 

    return model, weights.DEFAULT.transforms()

# Load pretrained EfficientNetV2 on ImageNet 
def load_efficientnet_from_tv(n_classes, freeze = True): 
    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = torchvision.models.get_model("efficientnet_v2_m", weights="DEFAULT")
    
    if freeze: 
        for param in model.parameters(): 
            param.requires_grad = False 
    
    num_ftrs = list(model.classifier.modules())[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=True), 
        nn.Linear(in_features=num_ftrs, out_features=n_classes)
    )
    for param in model.classifier.parameters(): 
        param.requires_grad = True 
    
    return model, weights.DEFAULT.transforms()

# Load pretrained VitT B 16 on ImageNet 
def load_vit_from_tv(n_classes, freeze = True): 
    weights = ViT_B_16_Weights.DEFAULT
    model = torchvision.models.get_model("vit_b_16", weights="DEFAULT")

    if freeze: 
        for param in model.parameters(): 
            param.requires_grad = False 
    
    num_ftrs = list(model.heads.modules())[-1].in_features
    model.heads[-1] = nn.Linear(in_features=num_ftrs, out_features=n_classes)
    for param in model.heads.parameters(): 
        param.requires_grad = True 
    
    return model, weights.DEFAULT.transforms()


# Load pretrained models on ImageNet from torchvision library 
def load_from_tv(modelname, n_classes, freeze=True): 
    if modelname[:6] == 'resnet': 
        model, preprocess = load_resnet_from_tv(modelname, n_classes, freeze=freeze)
    elif modelname[:len('googlenet')] == "googlenet": 
        model, preprocess = load_googlenet_from_tv(n_classes, freeze=freeze)
    elif modelname[:len("EfficientNetv2")] == "efficientnetv2": 
        model, preprocess = load_efficientnet_from_tv(n_classes, freeze=freeze)
    elif modelname[:len("ViT")] == "vit": 
        model, preprocess = load_vit_from_tv(n_classes, freeze=freeze)
    else: 
        model, preprocess = load_resnet_from_tv("resnet50", n_classes, freeze=freeze)
    return model, preprocess 