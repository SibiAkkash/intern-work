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

    print(f"{w}x{h} at {fps} fps")

    for frame_count in range(1, total_frames + 1):
        print(f"{frame_count = }")

        _, img = cap.read()

        img = cv2.resize(img, (540, 960))

        cv2.imshow("stream", img)

        if cv2.waitKey(0) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_frames(source: str, save_dir: str, frames, vid_name: str):
    cap = cv2.VideoCapture(source)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frame_in_list = max(frames)

    extension = "jpg"

    for frame in range(1, total_frames + 1):
        if frame > max_frame_in_list:
            break

        _, img = cap.read()

        if frame in frames:
            print(f"saving frame: {frame}")

            # resize to 360x640
            img = cv2.resize(img, (360, 640))

            cv2.imwrite(f"{save_dir}/{vid_name}_{frame:05d}.{extension}", img)

    cap.release()
    cv2.destroyAllWindows()


def save_frames_using_loader(source: str, save_dir: str):
    stream = LoadImages(path=source)

    for path, img, img0, vid_cap in stream:
        img = cv2.resize(img0, (540, 960))

        cv2.imshow("img", img)

        if cv2.waitKey(1) == ord("q"):
            break
def read_thread(source: str, save_path: str):
    pass


if __name__ == "__main__":
    # source = "http://localhost:8080/stream/video.mjpeg"
    source = "/home/sibi/Downloads/cycle_videos/rec_6_flip.mp4"

    frames = [
        1116,
        1128,
        1336,
        1341,
        1347,
        1353,
        1356,
        8660,
        8695,
        8735,
        8771,
        6140,
        7596,
        8962,
        8982,
        9023,
        9035,
        9058,
        10233,
        3413,
        3546,
        6185,
        6573,
        7715,
        7717,
        7726,
        7732,
        7750,
        7759,
        8791,
        8898,
        8771,
        8782,
        8932,
        9091,
        9089,
        10307,
        11305,
        11423,
        3743,
        5012,
        6475,
        7950,
        7956,
        10561,
    ]

    frames = set(frames)

    print(len(frames))

    save_frames(
        source=source, save_dir="experiments/new_imgs", frames=frames, vid_name="rec_6"
    )

    # save_frames_using_loader(source=source, save_dir=None)
    # view_frame_by_frame(source=source)
