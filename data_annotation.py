
import cv2
from utils.datasets import LoadStreams, LoadWebcam, LoadImages
import time
from typing import List
import imgaug as ia
import imgaug.augmenters as iaa


# Start webcam stream through uv4l
# uv4l -f --auto-video_nr --driver uvc --device-id 1224:2a25

def view_frame_by_frame(source: str):
    cap = cv2.VideoCapture(source)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'{w}x{h} at {fps} fps')

    for frame_count in range(1, total_frames + 1):
        print(f'{frame_count = }')
        
        _, img = cap.read()
        
        img = cv2.resize(img, (540, 960))

        cv2.imshow("stream", img)

        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_frames(source: str, save_dir: str, frames: List[int], vid_name: str):
    cap = cv2.VideoCapture(source)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extension = 'jpg'

    print(total_frames)

    for frame in range(1, total_frames + 1):
        _, img = cap.read()
        if frame in frames:
            print(f'saving frame: {frame}')

            # resize to 360x640
            img = cv2.resize(img, (360, 640))
            
            cv2.imwrite(f"{save_dir}/{vid_name}_{frame:04d}.{extension}", img)
        
    cap.release()
    cv2.destroyAllWindows()



def save_frames_using_loader(source: str, save_dir: str):
    stream = LoadImages(path=source)

    for path, img, img0, vid_cap in stream:
        img = cv2.resize(img0, (540, 960))
        
        cv2.imshow("img", img)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
def read_thread(source: str, save_path: str):
    pass


if __name__ == "__main__":
    # source = "http://localhost:8080/stream/video.mjpeg"
    source = "/home/sibi/Downloads/cycle_videos/rec_4.mp4"

    # frames = [
    #     269, 271, 272, 275, 279, 280, 283, 285, 290, 292, 297, 300, 303, 311, 313, 315,
    #     333, 341, 349, 352, 355, 379, 380, 381, 386, 1128, 1129, 1130, 1131, 1132, 1134, 
    #     1141, 1142, 1145, 1148, 1149, 1150, 1152, 1153, 1286, 1319, 1321, 1323, 1326, 1329, 
    #     1330, 1331, 1333, 1356, 1367, 128, 135, 152, 225, 254, 304, 345, 378, 705, 708, 
    #     795, 854, 895, 979, 1045, 1405, 3, 12, 17, 20, 27, 39, 46, 51, 52, 53, 54, 55, 56, 
    #     57, 58, 714, 715, 717, 722, 728, 732, 740, 742, 743, 754, 756
    # ]

    frames = [527, 1032, 1349, 4457]

    # frames = set(frames)

    print(len(frames))

    save_frames(source=source, save_dir="experiments/test_ds/val/images", frames=frames, vid_name="rec_4")

    # save_frames_using_loader(source=source, save_dir=None)
    # view_frame_by_frame(source=source)

    