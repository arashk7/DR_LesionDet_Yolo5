import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


source = 'E:\Dataset\DR\DeepDr\merged_tr_vl/55/55_l2.jpg'
weights = 'weights/exp15.pt'
view_img = True
save_txt = True
save_img = True
save_conf = True
imgsz = 640
device = 'cpu'
augment = True
conf_thres = 0.25
iou_thres = 0.45
save_dir = 'data/output'

classes = None
agnostic_nms = True
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

if half:
    model.half()  # to FP16




# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
t0 = time.time()

# Set Dataloader
dataset = LoadImages(source, img_size=imgsz)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

        # save_path = str(save_dir / p.name)
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n} {names[int(c)]}s, '  # add to string

            # Write results
            # for *xyxy, conf, cls in reversed(det):
            #     if save_txt:  # Write to file
            #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # if save_img or view_img:  # Add bbox to image
                #     label = f'{names[int(cls)]} {conf:.2f}'
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
