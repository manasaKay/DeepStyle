from torchvision import models

def our_model(device, model_name = 'vgg_19'):
    if model_name == 'vgg_19':
        model = models.vgg19(pretrained=True).features     
    elif model_name == 'vgg_16':
        model = models.vgg16(pretrained=True).features
    elif model_name == 'resnet':
        model = models.resnet18(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True).features
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model



    
