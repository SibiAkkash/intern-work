# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    python3 .\detect.py --source 0 --weights yolov5s.pt --conf 0.25 --view-img --class 65 67 76 --device 0
"""

from datetime import time
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.plots import Annotator, colors
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    check_suffix,
    colorstr,
    increment_path,
    is_ascii,
    non_max_suppression,
    print_args,
    save_one_box,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.datasets import LoadImages, LoadStreams
from models.experimental import attempt_load
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from frozendict import frozendict
from pprint import pprint
import mysql.connector

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


REMOTE_ID = 65
KEYBOARD_ID = 66
CELL_PHONE_ID = 67
BOOK_ID = 73
SCISSORS_ID = 76

DONE_COLOUR = (6, 201, 65)
NOT_DONE_COLOUR = (255, 255, 255)

X_OFFSET = 30
Y_OFFSET = 50
Y_PADDING = 80
RECT_POINT_1 = (X_OFFSET - 60, Y_OFFSET - 60)
RECT_POINT_2 = (800, Y_OFFSET + (Y_PADDING * 3) + 60)

db_connection_config = {
    "user": "sibi",
    "password": "pass1234",
    "host": "localhost",
    "database": "test_db"
}

# we get steps from db
# steps = list of labels (index of label is the step number)
# TODO assuming there cannot be duplicate labels (different steps cannot have same object)
# process_steps = [CELL_PHONE_ID, REMOTE_ID, SCISSORS_ID]
process_steps = [CELL_PHONE_ID, REMOTE_ID, SCISSORS_ID]
start_marker_object_id = KEYBOARD_ID
end_marker_object_id = BOOK_ID

NUM_STEPS = len(process_steps)

# ----------------- FLOW -------------------------------------------------------------------------------------
# start of cycle is presence of start marker
# start of step is presence of corresponding bounding box
# when we see a new bounding box, the previous step is over, calculate time taken
# if this new bounding box is the end marker, the cycle has finished, refresh state
# Step order doesnt matter, cycle start and end is based on presence of start and end marker respectively.
# ------------------------------------------------------------------------------------------------------------


state = {
    "prev_step": -1,
    "seen_objects": [],

    "num_steps_completed": 0,

    "cycle_started": False,
    "cycle_ended": False,
    "steps_started": False,
    "process_completed": False,

    # "state_changed": False,
    "cycle_start_frame_num": 0,
    "cycle_end_frame_num": 0,
    "step_start_frame_num": 0,
    "step_end_frame_num": 0,

    "is_step_completed": [False] * NUM_STEPS,
    "step_times": [0] * NUM_STEPS,
    "sequence": [],
}

def get_time_elapsed_ms(start_frame, end_frame, fps):
    return 1000.0 * (end_frame - start_frame) / fps


def get_step_number(detections, frame_num, total_frames, fps):
    """ if list contains objects we havent seen, return the step number and object id """
    # TODO this assumes we will only see 1 new object from the previous frame
    for object_id in detections:
        # return (step number, object_id) if we see a new object that we haven't seen yet
        if object_id not in state["seen_objects"] and object_id in process_steps:
            return process_steps.index(object_id), object_id

    return -1, -1
    
def handle_state_change(next_step_num, next_object_id, frame_num, fps, is_last_step=False):

    # check if this is the first object other than the start marker
    if len(state["seen_objects"]) == 1:
        print("FIRST OBJECT SEEN")
        # this means this is the first process object to be seen
        state["steps_started"] = True
        state["step_start_frame_num"] = frame_num
        state["prev_step"] = next_step_num
        state["seen_objects"].append(next_object_id)

    else:
        print("not first object")
        # calc step time for previous step
        prev_step = state["prev_step"]

        # previous step ends here
        state["step_end_frame_num"] = frame_num

        # calculate cycle time for the previous step
        time_taken = get_time_elapsed_ms(state["step_start_frame_num"], state["step_end_frame_num"], fps)
        print(f'Time taken to complete step {prev_step}: {time_taken} ms')
        state["step_times"][prev_step] = round(time_taken, 2)
        
        state["is_step_completed"][prev_step] = True
        state["sequence"].append(prev_step)
        state["num_steps_completed"] += 1

        if not is_last_step:
            # set start frame number for next step
            # if next_step_num <= NUM_STEPS and frame_num < total_frames:
            state["step_start_frame_num"] = frame_num + 1

            # set prev_step to the current step
            state["prev_step"] = next_step_num

            state["seen_objects"].append(next_object_id)

    
    pprint(state)




def is_object_present(detections, object_id):
    return object_id in detections


def handle_cycle_start(detections, frame_num):
    if start_marker_object_id in detections:
        state["cycle_started"] = True
        state["cycle_start_frame_num"] = frame_num
        state["seen_objects"].append(start_marker_object_id)
        print("CYCYLE STARTED")


def handle_cycle_end(frame_num, fps):
    if state["prev_step"] != -1:
        handle_state_change(frame_num=frame_num, fps=fps, is_last_step=True)
    
    state["cycle_ended"] = True
    state["cycle_ended_frame_num"] = frame_num
    print("CYCYLE ENDED")
    # TODO refresh state

# number of steps, step configuration is in db, make query to get data
def check_step(detections, frame_num, total_frames, fps):
    # test for one cycle now
    if state["cycle_ended"]:
        return

    # we have seen the start marker
    # the next frame, we dont have to check for cycle start again, it has started already
    if not state["cycle_started"]:
            if is_object_present(detections, start_marker_object_id):
                handle_cycle_start(detections, frame_num)
                pprint(state)

    if not state["cycle_started"]:
        return

    if is_object_present(detections, end_marker_object_id):
        handle_cycle_end(frame_num, fps)

    # cycle started, we see a object(the start marker could still be in frame) that is not the end marker
    # check if there is an object other than previously seen objects (including start marker)
    next_step_num, next_object_id = get_step_number(detections, frame_num, total_frames, fps)
    # print(type(next_object_id))
    # we havent seen a new object yet
    if next_step_num == -1:
        return

    print(f'next_step_num: {next_step_num}')

    handle_state_change(next_step_num, next_object_id, frame_num, fps, is_last_step=False)


def show_steps(image):
    # box background
    cv2.rectangle(
        img=image,
        color=(100, 100, 100),
        pt1=RECT_POINT_1,
        pt2=RECT_POINT_2,
        thickness=-1,
    )
    # steps status
    for step in range(NUM_STEPS):
        cv2.putText(
            img=image,
            text=f"Step {step + 1}, time_taken: {state['step_times'][step]} ms",
            org=(X_OFFSET, Y_OFFSET + Y_PADDING * (step)),
            fontFace=0,
            fontScale=1,
            color=DONE_COLOUR if state["is_step_completed"][step] else NOT_DONE_COLOUR,
            thickness=2,
        )


@torch.no_grad()
def run(
    weights="yolov5s.pt",  # model.pt path(s)
    source="data/images",  # file/dir/URL/glob, 0 for webcam
    imgsz=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
):

    save_img = not nosave and not source.endswith(".txt")  # save inference images
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = (
        False,
        Path(w).suffix.lower(),
        [".pt", ".onnx", ".tflite", ".pb", ""],
    )
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (
        suffix == x for x in suffixes
    )  # backend booleans
    stride, names = 64, [f"class{i}" for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name="resnet50", n=2)  # initialize
            modelc.load_state_dict(
                torch.load("resnet50.pt", map_location=device)["model"]
            ).to(device).eval()
    elif onnx:
        check_requirements(("onnx", "onnxruntime"))
        import onnxruntime

        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(("tensorflow>=2.4.1",))
        import tensorflow as tf

        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(
                    lambda: tf.compat.v1.import_graph_def(gd, name=""), []
                )  # wrapped import
                return x.prune(
                    tf.nest.map_structure(x.graph.as_graph_element, inputs),
                    tf.nest.map_structure(x.graph.as_graph_element, outputs),
                )

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, "rb").read())
            frozen_func = wrap_frozen_graph(
                gd=graph_def, inputs="x:0", outputs="Identity:0"
            )
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # is TFLite quantized uint8 model
            int8 = input_details[0]["dtype"] == np.uint8

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)


    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != "cpu":
        # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))

    dt, seen = [0.0, 0.0, 0.0], 0

    # process each image
    # self.sources, img, img0, None
    for path, img, im0s, vid_cap in dataset:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        t1 = time_sync()
        if onnx:
            img = img.astype("float32")
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize
                else False
            )
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(
                session.run(
                    [session.get_outputs()[0].name], {session.get_inputs()[0].name: img}
                )
            )
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]["quantization"]
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]["index"], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]["index"])
                if int8:
                    scale, zero_point = output_details[0]["quantization"]
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)
            """
            detection format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, class]
            pred = tensor([detection], [detection], ...)
            Output: 0: 480x640 1 person, 1 cell phone, Done. (0.595s)
            """
            # print(det)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            s += "%gx%g " % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # class_number: number of detections
                detections = {}
                det_list = []

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # num detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    detections[int(c.item())] = int(n.item())
                    det_list.append(int(c.item()))

                check_step(detections=det_list, frame_num=dataset.frame, total_frames=dataset.frames, fps=fps)
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy, conf, cls)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(
                                xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            
            # * show steps
            show_steps(im0)

            # Print time (inference-only)
            print(f"{s}Done. ({t3 - t2:.3f}s)")

            # Stream results
            im0 = annotator.result()

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(5)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    print(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    
    pprint(state)

    # write cycle times to db
    # cursor = cnx.cursor()
    # add_times = ("INSERT INTO emp_perf "
    #             "(emp_id, task_1_cycle_time_ms, task_2_cycle_time_ms, task_3_cycle_time_ms) "
    #             "VALUES (%s, %s, %s, %s)")
    
    # data_times = (2, *state["step_times"])
    # cursor.execute(add_times, data_times)
    # cnx.commit()
    # cursor.close()



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model path(s)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    # cnx = mysql.connector.connect(**db_connection_config)
    main(opt)
    # cnx.close()
