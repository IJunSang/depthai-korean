import cv2
import depthai as dai
from util.functions import non_max_suppression
import argparse
import time
import numpy as np
import sys
import os

def create_pipeline():
    pipeline = dai.Pipeline()

    nn = pipeline.createNeuralNetwork()
    # nn.setBlobPath('yolov5s_custom.blob')
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

    nn2 = pipeline.createNeuralNetwork()
    nn2.setBlobPath('yolov5s_licenseplate.blob')

    nn2.setNumPoolFrames(4)
    nn2.input.setBlocking(False)
    nn2.setNumInferenceThreads(2)
    
    detection_in_license = pipeline.createXLinkIn()
    detection_in_license.setStreamName("detection_in_license")
    detection_in_license.out.link(nn2.input)
    nn2_out = pipeline.createXLinkOut()
    nn2_out.setStreamName("nn2")
    nn2.out.link(nn2_out.input)

    return pipeline

files = ['img_inference/img_' + str(i) + '.jpg' for i in range(1, 11)]
labelMap = ["car", "licenseplate"]
labelMap = ["licenseplate"]

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

            # label = f"{labelMap[cls]}: {conf:.2f}"
            label = f"{conf:.2f}"
            color = colors[i, 0, :].tolist()

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            frame = cv2.rectangle(frame, (x1, y1 - 2*h), (x1+w, y1), (0, 0, 255), -1)
            frame = cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

def draw_boxes2(frame, boxes, total_classes):
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

            # label = f"{labelMap[cls]}: {conf:.2f}"
            label = f"{conf:.2f}"
            color = colors[i, 0, :].tolist()

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            frame = cv2.rectangle(frame, (x1, y1 - 2*h), (x1+w, y1), (255, 0, 0), -1)
            frame = cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


with dai.Device(create_pipeline()) as device:

    detIn = device.getInputQueue("detection_in")
    detIn_license = device.getInputQueue("detection_in_license")
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
    qDet2 = device.getOutputQueue("nn2", maxSize=4, blocking=False)

    images = os.listdir('img_inference')

    for index, image in enumerate(images):
        img = cv2.imread(os.path.join('./img_inference', image))
        img = cv2.resize(img, (416, 416))
        img = img.transpose(2, 0, 1)
        lic_frame = dai.ImgFrame()
        
        lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
        
        lic_frame.setData(img)
        
        lic_frame.setWidth(416)
        lic_frame.setHeight(416)
        
        frame = lic_frame.getCvFrame()
        
        detIn.send(lic_frame)
        detIn_license.send(lic_frame)

        in_nn = qDet.get()
        in_nn2 = qDet2.get()

        layers = in_nn.getAllLayers()
        layers2 = in_nn2.getAllLayers()

        output = np.array(in_nn.getLayerFp16("output"))
        output2 = np.array(in_nn2.getLayerFp16("output"))

        cols = output.shape[0]//10647
        cols2 = output2.shape[0]//10647
        output = np.reshape(output, (10647, cols))
        output = np.expand_dims(output, axis=0)
        output2 = np.reshape(output2, (10647, cols2))
        output2 = np.expand_dims(output2, axis=0)

        total_classes = cols - 5
        total_classes2 = cols2 - 5

        boxes = non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
        boxes = np.array(boxes[0])
        boxes2 = non_max_suppression(output2, conf_thres=0.3, iou_thres=0.4)
        boxes2 = np.array(boxes2[0])

        if boxes is not None:
            frame = draw_boxes(frame, boxes, total_classes)
        if boxes2 is not None:
            frame = draw_boxes2(frame, boxes2, total_classes2)

        cv2.imwrite("result_img_"+str(index)+'.jpg', frame)
        

        