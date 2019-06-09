from PIL import Image
from models import *
from utils import *
import cv2
import torch
import skimage.io


def test_():
    print ("Hi!")

    
def train_model(style_path, content_path, alpha, beta, imshow_flag, graph_flag, num_epochs, weights, optimiser = 'Adam', model_name = 'vgg_19', avg_flag = 0, scale_percent = 100, rotate_percent = 0, multiple_style_flag = 0, style_path_2 = '/data/style_images/candy.jpg', interp_value = (0.5, 0.5), preserve_color = 0, show_iter=0):    
    device = gpu_check()
 
    img1 = Image.open(content_path).convert('RGB')    # img1 is content
    img2 = Image.open(style_path).convert('RGB')      # img2 is style
    
    if scale_percent != 100:
        img2 = cv2.imread(style_path, cv2.IMREAD_UNCHANGED)
        scale_percent = scale_percent 
        width = int(img2.shape[1] * scale_percent / 100)
        height = int(img2.shape[0] * scale_percent / 100)
        dim = (width, height)
        img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_NEAREST)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) 
        img2 = Image.fromarray(img2, 'RGB')

    if rotate_percent != 0:
        img2 = img2.rotate(rotate_percent)
    
    img1_before = img1
    img2_before = img2

    net = our_model(device, model_name)
    img1, img2 = transformation(img1).to(device), transformation(img2).to(device)
    img1_features, img2_features = get_features(img1, net, model_name, avg_flag), get_features(img2, net, model_name, avg_flag) 
    correlations = {l: correlation_matrix(img2_features[l]) for l in img2_features}
    
    if multiple_style_flag == 1:
        img3 = Image.open(style_path_2).convert('RGB')   
        img3_before = img3
        img3 = transformation(img3).to(device)
        img3_features = get_features(img3, net, model_name, avg_flag)
        correlations_2 = {l: correlation_matrix(img3_features[l]) for l in img3_features}
        
    target = img1.clone().requires_grad_(True).to(device) 
    
    if optimiser == 'Adam':
        optimizer = torch.optim.Adam([target], lr=0.09)
    elif optimiser == 'Adagrad':
        optimizer = torch.optim.Adagrad([target], lr=0.09)
    elif optimiser == 'SGD':
        optimizer = torch.optim.SGD([target], lr=0.09, momentum=0.9)
    elif optimiser == 'RMSProp':
        optimizer = torch.optim.RMSprop([target], lr=0.09)
    elif optimiser == 'LBFGS':
        optimizer = torch.optim.LBFGS([target], lr=0.09)
         
    content_loss_list, style_loss_list, total_loss_list, epochs = [], [], [], []
    for count in range(0, num_epochs):
        target_features = get_features(target, net, model_name, avg_flag)      
        def closure():
            optimizer.zero_grad()
            content_loss, style_loss = 0, 0
            for layer in weights: 
                loss = target_features[layer] - img1_features[layer]
                content_loss  += ( torch.mean((loss)**2) * weights[layer] )  
            for layer in weights:
                target_feature = target_features[layer]
                target_corr = correlation_matrix(target_feature)
                style_corr = correlations[layer]
                layer_loss = torch.mean((target_corr - style_corr)**2)
                layer_loss *= weights[layer]
                if multiple_style_flag == 1:
                    style_corr_2 = correlations_2[layer]
                    layer_loss_2 = torch.mean((target_corr - style_corr_2)**2)
                    layer_loss_2 *= weights[layer]
                _, d, h, w = target_feature.shape
                style_loss += (interp_value[0] * layer_loss / (d * h * w))
                if multiple_style_flag == 1:
                    style_loss += (interp_value[1] * layer_loss_2 / (d * h * w)) 
            total_loss = alpha * style_loss + beta * content_loss
            content_loss_list.append(content_loss.item())
            style_loss_list.append(style_loss.item())
            total_loss_list.append(total_loss.item())
            epochs.append(count)
            total_loss.backward(retain_graph=True)
            return total_loss
        
        if optimiser == 'LBFGS':
            optimizer.step(closure)
        else: 
            closure()
            optimizer.step()
            
        if show_iter==1 and count%300==0:
            print (count)
            imshow_(img1_before, img2_before, target)
            
    if imshow_flag == 1:
        imshow_(img1_before, img2_before, target)
        if multiple_style_flag == 1:
            imshow_(img1_before, img3_before, target)
            
    if graph_flag == 1:
        plot_graph(content_loss_list, style_loss_list, total_loss_list, epochs)