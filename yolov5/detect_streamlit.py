import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import streamlit as st


class ADetect:
    def detect(self, img, conf_level=0.29):
        '''arash'''
        weights = 'weights/exp15.pt'
        imgsize = 640
        devicee = ''
        webcam = False
        conf_thres = conf_level
        iou_thres = 0.45
        agnostic_nms = False
        classes = None
        augment = False
        source = 'D:\dataset\F_Yolo/train/images/0a3de3f0-10f2-415a-968a-377450d4eb4a.jpg'

        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(devicee)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsize, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img1 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img1.half() if half else img1) if device.type != 'cpu' else None  # run once
        s = ''

        # img = Image.open(source)#uploaded_file.stream
        # img = cv2.imread(uploaded_file.stream)
        # img = Image.open('E:\Dataset\DR\DeepDr\merged_tr_vl/66/66_l2.jpg')
        # img = img.resize((imgsz, imgsz))
        # img = np.asarray(img)
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # img = np.reshape(img, (3, imgsz, imgsz))

        img0 = img.copy()
        # Padded resize
        img = letterbox(img, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        for path, imgo, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                                       agnostic=agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = Path(path), '', img0, getattr(dataset, 'frame', 0)

                # a = getattr(dataset, 'frame', 0)
                # save_path = str(save_dir / p.name)
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                lesions = []
                counter = {}
                lesion_list = {}
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string
                        counter[names[int(c)]] = f'{n}'

                    # dict of results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        n = counter[f'{names[int(cls)]}']
                        params = {'cls': f'{names[int(cls)]}', 'n': counter[f'{names[int(cls)]}'], 'x': xywh[0],
                                  'y': xywh[1], 'w': xywh[2],
                                  'h': xywh[3]}
                        lesions.append(params)
                        # xyxy[0]-=105
                        # xyxy[2] -= 105
                        # xyxy[1] -= 80
                        # xyxy[3]-=80
                        plot_one_box(xyxy, img0, label=f'{names[int(cls)]}', color=colors[int(cls)], line_thickness=3)

                    # for les in lesions:
                    #      lesion_list[f'{names[int(cls)]}']+=les[]
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #
                    #     if save_img or view_img:  # Add bbox to image
                    #         label = f'{names[int(cls)]} {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                return lesions, img0


def lesion_detection():
    ''' Hide Menu and Logo'''
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    sel = st.selectbox("Choose Fundus Image:", ('Sample 1', 'Sample 2', 'Sample 3', 'Upload Image File'))

    det = ADetect()
    img = 0
    if sel == 'Sample 1':
        img = cv2.imread('static/media/8_r2.jpg')
    if sel == 'Sample 2':
        img = cv2.imread('static/media/8_l1.jpg')
    if sel == 'Sample 3':
        img = cv2.imread('static/media/20_r1.jpg')

    if sel == 'Upload Image File':
        img = cv2.imread('static/media/20_r1.jpg')
        uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])

        if uploaded_file is not None:
            # file_details = {"FileName": 'static/media/'+uploaded_file.name, "FileType": uploaded_file.type,
            #                 "FileSize": uploaded_file.size}
            # st.write(file_details)
            # img = cv2.imread('static/media/'+uploaded_file.name)
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

    if img is not None:
        st.image(img, use_column_width=True, channels="BGR")

        conf_level = st.slider('Confidence Threshold', min_value=0.1, max_value=0.99, value=0.29)
        st.write('Value:', conf_level)
        lesions, img = det.detect(img, conf_level)

        st.image(img, use_column_width=True, channels="BGR")
    # st.subheader(lesions)


"""
# Zhitang - Fundus Image Analysis
by Ahmad Karambakhsh (arash)
"""
selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Lesion Detection', 'DR Grading', 'Quality Evaluation')
)

if selected_box == 'Lesion Detection':
    lesion_detection()
if selected_box == 'DR Grading':
    st.subheader("In Progress")
if selected_box == 'Quality Evaluation':
    st.subheader("In Progress")
