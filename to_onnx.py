import torch
import torch.onnx
from yolo import YOLO


yolo = YOLO()
net = yolo.generate()
 

#data type nchw
dummy_input1 = torch.randn(1, 3, 608, 608)
input_names = [ "image_input"]
output_names = [ "output" ]
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(net, dummy_input1, "yolov4_coco.onnx", verbose=True, input_names=input_names, output_names=output_names)
