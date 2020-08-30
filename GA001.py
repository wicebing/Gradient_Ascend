import pandas as pd
import numpy as np
from PIL import Image
import math 
import glob
import os
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T
from torchvision import models


lr = 1e-1
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

def show_pic(image,boxs,labels,n=0):
    display_image = np.array(image)
    for i, bbox in enumerate(boxs):
        display_image = vis_bbox(display_image, bbox)
        display_image = vis_class(display_image, bbox, _COCO_INSTANCE_CATEGORY_NAMES[labels[i]])
    
    plt.figure(figsize=(10, 10),dpi=200)
    plt.imshow(display_image)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'./results/test{n}.png', bbox_inches="tight")
    
transform = T.ToTensor()
image_path = './results/AA_Resnet.png'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.size())

# using Mask RCNN
detector = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
detector.to(device)

# for granding ascend
image_tensor = image_tensor.to(device)
image_tensor.requires_grad = True

for i in range(3):
    detector.eval()
    
    results = detector(image_tensor)
    
    target_labels = results[0]['labels'][results[0]['labels']>1]
    target_boxes = results[0]['boxes'][results[0]['labels']>1]
    target_masks = results[0]['masks'][results[0]['labels']>1]
    
    target = [{'boxes':target_boxes,
               'labels':target_labels,
               'masks':target_masks
               }]
    
    # pic = image_tensor.permute(0,2,3,1).detach().numpy()[0]
    # reT = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    #                           T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    #                           T.ToPILImage()])
    
    if i%10000000==0:
        pic = image_tensor[0].cpu()
        reT = T.Compose([T.ToPILImage()])
    
        REV_pic = reT(pic)
        REV_pic.save('./results/AA_Resnet.png')
        
        display_pic= np.array(REV_pic)
        
        show_pic(display_pic,target_boxes,target_labels,n=i)
        print(i)
    
    # calculate the grading
    detector.train()
    # zero_grad
    
    loss = detector(image_tensor,target)
    
    total_loss = 0
    
    for l in loss.values():
        total_loss += l
    total_loss.backward()
    
    with torch.no_grad():
        image_tensor -= lr*image_tensor.grad

