import pandas as pd
import numpy as np
from PIL import Image
import math 
import glob
import os
import matplotlib.pyplot as plt
import cv2
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T
from torchvision import models


lr = 1
epoch = 10000000
batch = 1
show_bs = 100
fit = 200

device = 'cuda:0'

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

_COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def vis_bbox(image, bbox, color=_GREEN, thick=1):
    """Visualizes a bounding box."""
    image = image.astype(np.uint8)
    bbox = list(map(int, bbox))
    x0, y0, x1, y1 = bbox
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thick)
    return image

def vis_class(image, bbox, text, bg_color=_GREEN, text_color=_GRAY, font_scale=0.35):
    """Visualizes the class."""
    image = image.astype(np.uint8)
    x0, y0 = int(bbox[0]), int(bbox[1])

    # Compute text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)

    # Place text background
    back_tl = x0, y0 - int(1.3 * text_h)
    back_br = x0 + text_w, y0
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)

    # Show text
    text_tl = x0, y0 - int(0.3 * text_h)
    cv2.putText(image, text, text_tl, font, font_scale,
                text_color, lineType=cv2.LINE_AA)

    return image

def vis_mask(image, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    image = image.astype(np.float32)

    mask = mask >= 0.5
    mask = mask.astype(np.uint8)
    idx = np.nonzero(mask)

    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += alpha * col

    if show_border:
        contours = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(image, contours, -1, _WHITE,
                         border_thick, cv2.LINE_AA)

    return image.astype(np.uint8)

def show_pic(image,boxs,labels,masks,n=0,is_mask=False):
    display_image = np.array(image)
    masks = np.squeeze(masks.detach().cpu().numpy(), axis=1)
    for i, bbox in enumerate(boxs):
        display_image = vis_bbox(display_image, bbox)
        display_image = vis_class(display_image, bbox, _COCO_INSTANCE_CATEGORY_NAMES[labels[i]])
        
        if is_mask:
            display_image = vis_mask(display_image, masks[i], np.array([0., 0., 255.]))
    
    plt.figure(figsize=(10, 10),dpi=200)
    plt.xticks([])
    plt.yticks([])
    plt.imsave(f'./results/test{n}.png',display_image)

    # plt.savefig(f'./results/test{n}.png', bbox_inches="tight")

def iou(box_pred,box_targ):
    #input size = [box_n,5]  target-1-to-all-predict
    box_n=box_targ.shape[0]
    #box_target num = box_n 
    #recover to normal size ratio
    area_pred = torch.prod(box_pred[:,[2,3]],1) #w*h
    area_targ = torch.prod(box_targ[:,[2,3]],1) #w*h
    Diff = torch.abs(box_pred-box_targ)
    Summ = (box_pred+box_targ)
    
    Status_x = Diff[:,0]>Diff[:,2]
    Status_y = Diff[:,1]>Diff[:,3]
    
    Ix = torch.zeros(box_n).float().to(device)
    Iy = torch.zeros(box_n).float().to(device)
    
    if (~Status_x).sum()>0:
        Ix[~Status_x]=(torch.min(box_pred[:,2],box_targ[:,2]))[~Status_x]
    if (~Status_y).sum()>0:
        Iy[~Status_y]=(torch.min(box_pred[:,3],box_targ[:,3]))[~Status_y]
    if (Status_x).sum()>0:
        Ix[Status_x]=(torch.clamp(0.5*Summ[:,2]-Diff[:,0],0))[Status_x]
    if (Status_y).sum()>0:
        Iy[Status_y]=(torch.clamp(0.5*Summ[:,3]-Diff[:,1],0))[Status_y]

    area_Inter = Ix*Iy
    out_IOU = area_Inter/(area_pred+area_targ-area_Inter+1e-6)

    return out_IOU     
    
def NMS(class_box, class_label, class_mask, IOU_T=0.3):
    box_n=class_box.shape[0]
    
    box_num = -1*torch.arange(box_n)
    outbox = []
    outlabel = []
    outmask = []
    while torch.sum(box_num>-999)>0:
        #select prob max one as select_box
        idx = torch.argmax(box_num)
        outbox.append(class_box[idx])
        outlabel.append(class_label[idx])
        outmask.append(class_mask[idx])
        
        select_box = class_box[idx].repeat(box_n).view(box_n,-1)
        #got_iou
        select_iou = iou(class_box,select_box)
        #iou_filter
        mask_iou_all = select_iou > 0.8
        mask_iou = select_iou > IOU_T
        mask_label = class_label==class_label[idx]
        #delete all > IOU_threshold including the selected box
        box_num[mask_iou_all | (mask_iou & mask_label)] =-999
    return torch.stack(outbox) , torch.stack(outlabel) , torch.stack(outmask)
    
transform = T.ToTensor()

try:
    image_path = './results/AA_Resnet.png'
    image = Image.open(image_path).convert("RGB")
except:
    image_path = './data/AA_Resnet.jpg'
    print(' *** use origin picture *** ')
    image = Image.open(image_path).convert("RGB")   
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.size())

image_path2 = './data/AA_Resnet.jpg'
image2 = Image.open(image_path2).convert("RGB")  
image_tensor2 = transform(image2)
image_tensor2 = image_tensor2.unsqueeze(0)

image_path3 = './results/AA_mask2.png'
image3 = Image.open(image_path3).convert("RGB")  
image_tensor3 = transform(image3)
image_tensor3 = image_tensor3.unsqueeze(0)

# using Mask RCNN
detector = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
detector.to(device)

# for granding ascend
image_tensor = image_tensor.to(device)
image_tensor.requires_grad = True
image_tensor_origin = image_tensor.clone()

bing_mask3 = torch.zeros([3,638,638],dtype=torch.bool)

t0 = time.time()

detector.eval()
image_tensor2 = image_tensor2.to(device)
image_tensor3 = image_tensor3.to(device)
results2 = detector(image_tensor2)
results3 = detector(image_tensor3)

bing_mask_01 = results2[0]['masks'][results2[0]['labels']==1][1]>0.5
# bing_mask_17 = results2[0]['masks'][results2[0]['labels']==1][1]>0.5

# bing_mask_001 = results3[0]['masks'][1]>0.5
bing_mask_71 = results3[0]['masks'][71]>0.5
bing_mask = bing_mask_01 | bing_mask_71

bing_mask3 = bing_mask.expand([3,638,638])

for i in range(epoch):
    detector.eval()   
    results = detector(image_tensor)
       
    # if i ==0:
    #     bing_mask_01 = results2[0]['masks'][results2[0]['labels']==1][1]>0.5
    #     # bing_mask_17 = results2[0]['masks'][results2[0]['labels']==1][1]>0.5        
    #     bing_mask_001 = results[0]['masks'][1]>0.5
    #     bing_mask_71 = results3[0]['masks'][71]>0.5
    #     bing_mask = bing_mask_01 | bing_mask_001 | bing_mask_71        
    #     bing_mask3 = bing_mask.expand([3,638,638])
    
    # for s in range(len(results[0]['labels'])):
        
    if i ==0 or i%batch==0:
        with torch.no_grad():
            target_labels = results[0]['labels']
            target_boxes = results[0]['boxes']
            target_masks = results[0]['masks']
            target_boxes , target_labels , target_masks = NMS(target_boxes, target_labels, target_masks, IOU_T=0.3)
            
            # # target_labels = results[0]['labels'][results[0]['labels']>1][:fit]
            # # target_boxes = results[0]['boxes'][results[0]['labels']>1][:fit]
            # # target_masks = results[0]['masks'][results[0]['labels']>1][:fit]
            
            
            # target_labels = results[0]['labels'][s:s+1]
            # target_boxes = results[0]['boxes'][s:s+1]
            # target_masks = results[0]['masks'][s:s+1]
            
            target = [{'boxes':target_boxes,
                       'labels':target_labels,
                       'masks':target_masks
                       }]

    if i%show_bs==0:
        pic = image_tensor[0].cpu()
        
        reT = T.Compose([T.ToPILImage()])
    
        REV_pic = reT(pic)
        REV_pic.save('./results/AA_Resnet.png')
        
        display_pic= np.array(REV_pic)
        
        show_pic(display_pic,target_boxes,target_labels,target_masks,n=i,is_mask=True)
        print(i)
        
        print(f'remain time: {epoch*(time.time()-t0)/(60*i+1)}')
    
    # pic = image_tensor.permute(0,2,3,1).detach().numpy()[0]
    # reT = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    #                           T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    #                           T.ToPILImage()]) 
    
    # calculate the grading
    detector.train()
    detector.zero_grad()
    
    loss = detector(image_tensor,target)
    
    total_loss = loss['loss_classifier']+loss['loss_mask']+loss['loss_box_reg']
    
    total_loss.backward()
    
    with torch.no_grad():
        image_tensor -= lr*image_tensor.grad
        image_tensor[0][bing_mask3] = image_tensor_origin[0][bing_mask3]
    # zero_grad
    image_tensor.grad.zero_()
