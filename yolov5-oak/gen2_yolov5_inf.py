import cv2
import depthai as dai
from util.functions import non_max_suppression
import argparse
import time
import numpy as np
import sys

def create_pipeline():
    pipeline = dai.Pipeline()

    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath('yolov5s_car.blob')

    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)

    detection_in = pipeline.createXLinkIn()
    detection_in.setStreamName("detection_in")

    detection_in.out.link(nn.input)

    nn_out = pipeline.createXLinkOut()
    nn_out.setStreamName("nn")
    nn.out.link(nn_out.input)

    return pipeline

files = ['img_inference/img_' + str(i) + '.jpg' for i in range(1, 11)]
labelMap = ["car", "pedestrian"]

def draw_boxes(frame, boxes, total_classes):
    if boxes.ndim == 0:
        return frame
    else:
        colors = boxes[:, 5] * (255 / total_classes)
        colors = colors.astype(np.uint8)
        colors = cv2.applyColorMap(colors, cv2.COLORMAP_HSV)
        colors = np.array(colors)

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
            conf, cls = boxes[i, 4], int(boxes[i, 5])

            label = f"{labelMap[cls]}: {conf:.2f}"
            color = colors[i, 0, :].tolist()

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            frame = cv2.rectangle(frame, (x1, y1 - 2*h), (x1+w, y1), color, -1)
            frame = cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


with dai.Device(create_pipeline()) as device:

    detIn = device.getInputQueue("detection_in")
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)

    for i in range(10):
        img = cv2.imread(files[i])
        img = cv2.resize(img, (1920, 1080))
        img = img.transpose(2, 0, 1)
        lic_frame = dai.ImgFrame()
        print(sys.getsizeof(lic_frame))
        lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
        print(sys.getsizeof(lic_frame))
        lic_frame.setData(img)
        print(sys.getsizeof(lic_frame))
        lic_frame.setWidth(1920)
        lic_frame.setHeight(1080)
        print(sys.getsizeof(lic_frame))
        frame = lic_frame.getCvFrame()
        print(sys.getsizeof(lic_frame))
        detIn.send(lic_frame)

        in_nn = qDet.get()

        layers = in_nn.getAllLayers()

        output = np.array(in_nn.getLayerFp16("output"))

        cols = output.shape[0]//10647
        output = np.reshape(output, (10647, cols))
        output = np.expand_dims(output, axis=0)

        total_classes = cols - 5

        boxes = non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
        boxes = np.array(boxes[0])

        if boxes is not None:
            frame = draw_boxes(frame, boxes, total_classes)

        cv2.imwrite("result_img_"+str(i)+'.jpg', frame)
        

        