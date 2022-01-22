import torch
import cv2
# from utils.datasets import LoadImages, LoadStreams
# from utils.general import set_logging
# from utils.torch_utils import select_device

print('loading model...')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5m.pt')
print('model loaded ')

source = '/home/sibi/Downloads/cycle_videos/rec_4.mp4'
img = 'experiments/test_ds/images/val/rec_4_0527.jpg'

results = model(img)
results.print()

print(dir(results))

print(results.tolist())

print(results.pred)


cap = cv2.VideoCapture(source)

while cap.isOpened():
    success, img = cap.read()

    if not success:
        continue

    img = cv2.resize(img, (540, 960))
    
    results = model(img)
    results.print()
    
    # TODO          very bad, saves every image in /tmp, opens new image
    # TODO          each time results.show() is called
    # results.show()

    print(results.pred)

    cv2.imshow("vid", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# pt = True
# stride = int(model.stride.max())
# imgsz = (640, 384)
# stream = LoadStreams(sources=source, img_size=640, stride=stride)

# set_logging()

# half = False

# device = select_device('0')
# half &= device.type != "cpu"

# if half:
#     model.half()  # to FP16

# if pt and device.type != "cpu":
#     # run once
#     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))    


# stream = LoadStreams(sources=source, img_size=640, stride=stride)
# stream = LoadImages(sources=source, img_size=640, stride=stride)
# bs = len(stream)

# for path, img , img0, vid_cap in stream:
    # results = model(img)
    # results.print()

    # cv2.imshow("vid", img0)
    # cv2.waitKey(5)


