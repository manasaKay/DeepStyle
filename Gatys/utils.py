from torchvision import transforms as tf
from torchvision import models
import cv2
import torch
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

# GPU check Utils
def gpu_check():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device


# All Image utils
def transformation(img):
    tasks = tf.Compose([tf.Resize(400), tf.ToTensor(), tf.Normalize((0.44,0.44,0.44),(0.22,0.22,0.22))])
    img = tasks(img)[:3,:,:].unsqueeze(0)    
    return img

def get_features(image, model, model_name, avg_flag = 0):
    if model_name == 'vgg_19':
        layers = {'0': 'conv1_1', '5': 'conv2_1',  '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        x = image
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x      
    
    if model_name == 'vgg_16':
        layers = {'0': 'conv1_1', '5': 'conv2_1',  '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        x = image
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x      
                
    if model_name == 'resnet':
        layers = set(['conv1','bn1','relu','maxpool','layer1','layer2', 'layer3', 'layer4'])
        x = image
        features = {}
        for name, layer in model._modules.items():
            if name == 'fc':
                x = x.view(-1, 512)
            x = layer(x)
            if name in layers:
                features[name] = x      
    
    if model_name == 'alexnet':
        layers = {'0': 'conv1', '3': 'conv2',  '6': 'conv3', '8': 'conv4', '10': 'conv5'}
        x = image
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x    
                
    if model_name == 'vgg_19' and avg_flag == 1:
        layers = {'0': 'conv1_1', '5': 'conv2_1',  '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        x = image
        features = {}
        for name, layer in model._modules.items():
            if (type(layer).__name__ == 'MaxPool2d'):
                layer = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            x = layer(x)
            if name in layers:
                features[layers[name]] = x   
                        
    return features

def tensor_to_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image *= np.array((0.22, 0.22, 0.22)) + np.array((0.44, 0.44, 0.44))
    image = image.clip(0, 1)
    return image

def imshow_(img1, img2, target):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.set_title('Content Image')
    ax1.imshow(img1)
    ax2.set_title('Style Image')
    ax2.imshow(img2)
    ax3.set_title('Mixed Image')
    ax3.imshow(tensor_to_image(target))
    
    
# Correlation Matrix utils
def correlation_matrix(tensor):
    _, d, h, w = tensor.size()    
    tensor = tensor.view(d, h * w)    
    correlation = torch.mm(tensor, tensor.t())
    return correlation


# All graph utils
def plot_graph(content_loss_list, style_loss_list, total_loss_list, epochs):  
    plt.figure()
    plt.plot( epochs, total_loss_list, color='red', label="Total loss")
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    return
