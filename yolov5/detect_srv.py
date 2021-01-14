import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def image_from_buffer(buffer):
    '''
    If we don't save the file locally and just want to open
    a POST'd file. This is what we use.
    '''
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    flag = 1
    # flag = 1 == cv2.IMREAD_COLOR
    # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
    frame = cv2.imdecode(bytes_as_np_array, flag)
    return frame

from PIL import Image, ImageOps
from flask import Flask, send_file, jsonify, render_template, make_response,redirect,url_for
from flask_restx import Api, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage


app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

parser.add_argument('metric')
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)

@api.route('/upload')
class upload(Resource):
    def get(self):
        return make_response(render_template("visualize.html"))

# @api.route('/uploader')
# class vis(Resource):
#     def get(self):
#         return make_response(render_template("upload.html"))
#
# @app.route('/display/<filename>')
# def display_image(filename):
# 	#print('display_image filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@api.expect(parser)
class Process(Resource):
    def post(self):
        args = parser.parse_args()
        uploaded_file = args['file']

        # torch.no_grad()

        # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://'))

        '''arash'''
        weights = 'weights/exp15.pt'
        imgsize = 640
        devicee = ''
        webcam = False
        conf_thres = 0.29
        iou_thres = 0.45
        agnostic_nms = False
        classes = None
        augment = False
        source = 'E:\Dataset\DR\DeepDr\merged_tr_vl/55/55_l2.jpg'

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
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        s = ''

        # img = Image.open(source)#uploaded_file.stream
        # img = cv2.imread(uploaded_file.stream)
        # img = Image.open('E:\Dataset\DR\DeepDr\merged_tr_vl/66/66_l2.jpg')
        # img = img.resize((imgsz, imgsz))
        # img = np.asarray(img)
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = image_from_buffer(uploaded_file)

        # img = np.reshape(img, (3, imgsz, imgsz))

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
                    p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

                # a = getattr(dataset, 'frame', 0)
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
                    #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #
                    #     if save_img or view_img:  # Add bbox to image
                    #         label = f'{names[int(cls)]} {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

        #         # Stream results
        #         if view_img:
        #             cv2.imshow(str(p), im0)
        #             if cv2.waitKey(1) == ord('q'):  # q to quit
        #                 raise StopIteration
        #
        #         # Save results (image with detections)
        #         if save_img:
        #             if dataset.mode == 'image':
        #                 cv2.imwrite(save_path, im0)
        #             else:  # 'video'
        #                 if vid_path != save_path:  # new video
        #                     vid_path = save_path
        #                     if isinstance(vid_writer, cv2.VideoWriter):
        #                         vid_writer.release()  # release previous video writer
        #
        #                     fourcc = 'mp4v'  # output video codec
        #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #                 vid_writer.write(im0)
        #
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")
        #
        # print(f'Done. ({time.time() - t0:.3f}s)')
        return s


#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     opt = parser.parse_args()
#     print(opt)

# with torch.no_grad():
#     if opt.update:  # update all models (to fix SourceChangeWarning)
#         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
#             detect()
#             strip_optimizer(opt.weights)
#     else:
#         detect()

api.add_resource(Process, '/process')

if __name__ == "__main__":
    app.run(debug=True)
