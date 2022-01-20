import cv2
from utils.datasets import LoadStreams, LoadWebcam
import time

# uv4l -f --auto-video_nr --driver uvc --device-id 1224:2a25

# source = "udp://0.0.0.0:8000/video.mjpeg"
source = "http://localhost:8080/stream/video.mjpeg"

# * LoadStreams isnt working
# stream = LoadStreams(sources=source)
stream = LoadWebcam(pipe=source)

for img_path, img, img0, _ in stream:
    cv2.imshow("vid", img0)

# for path, img, im0s, vid_cap in stream:
#     print(img.shape, type(img))
#     cv2.imshow("stream", img)
#     cv2.waitKey(5)

def opencv_read_stream():
    cap = cv2.VideoCapture(source)

    # Get each frame from  stream
    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'
            continue

        cv2.imshow("vid", img)

        if cv2.waitKey(5) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




def read_blocking_webcam_rotate(source: str, save_path: str):
    cap = cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'{w}x{h} at {fps} fps')
    # rotating video counter 90 deg clockwise
    codec = 'mp4'
    # filename: Input video file
    # fourcc: 4-character code of codec used to compress the frames
    # fps: framerate of videostream
    # framesize: Height and width of frame
    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*codec), fps, (h, w))

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            return
        
        # resize img
        img = cv2.resize(img, (960, 540))
        # rotate img
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # vid_writer.write(img)
        cv2.imshow("stream", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # opencv_read_stream()
    pass