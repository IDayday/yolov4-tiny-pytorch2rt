from torch._C import float16
from torch2trt import torch2trt
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from nets.yolo4_tiny import YoloBody
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)


model_path = 'model_data/yolov4_tiny_weights_coco.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create some regular pytorch model...
net = YoloBody(3, 80)
state_dict = torch.load(model_path, map_location=device)
net.load_state_dict(state_dict)
net = net.eval().cuda()
 
# create example data
x = torch.ones((1, 3, 416, 416)).cuda()
 
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(net, [x], fp16_mode = True)
torch.save(model_trt,'yolov4-tiny-coco-fp16.pth')
print('pth to tensorRT is OK!')