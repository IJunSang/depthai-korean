import cv2
import depthai as dai
from util.functions import non_max_suppression
import numpy as np
import sys
import os
import time

labelMap = ["Person"]
input_shape = 640

def create_pipeline():
    pipeline = dai.Pipeline()

    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath('water.blob')
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

def split_images(image):

    count = 0

    if image.shape == (1080, 1920, 3):

        crop = np.empty((6, input_shape, input_shape, 3))
        overlap = 440

        for i in range(1, 3):
            for j in range(1, 4):
                if i == 1:
                    crop[count] = image[0:input_shape, ((j-1)*input_shape):(j*input_shape), :]
                else:
                    crop[count] = image[overlap:(overlap+input_shape),((j-1)*input_shape):(j*input_shape),:]
                count += 1
    else:

        crop = np.empty((4, input_shape, input_shape, 3))
        overlap = 384
        
        for i in range(1, 3):
            for j in range(1, 3):
                if i == 1:
                    crop[count] = image[0:input_shape, ((j-1)*input_shape):(j*input_shape), :]
                elif i == 2:
                    crop[count] = image[overlap:(overlap+input_shape),((j-1)*input_shape):(j*input_shape),:]
                count += 1

    return crop

def get_textbox_size(label):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    return (w, h)

def draw_inf_time(image, w, h, label, index):
    if image.shape == (1080, 1920, 3):
        for i in range(2):
            for j in range(3):
                if i == 0:
                    image = cv2.rectangle(image, (input_shape * j, 0), (input_shape * (j + 1), input_shape), (0, 255, 255), 2)
                else:
                    image = cv2.rectangle(image, (input_shape * j, 440), (input_shape * (j + 1), 1080), (0, 255, 255), 2)
        if index < 3:
            image = cv2.rectangle(image, (input_shape * index, 0), (((input_shape * index) + (w*2)), (h*2)), (0, 0, 255), -1)
            image = cv2.putText(image, label, ((input_shape * index) + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            image = cv2.rectangle(image, (input_shape * index, 440), (((input_shape * index) + (w*2)), (440 + (h * 2))), (0, 0, 255), -1)
            image = cv2.putText(image, label, ((input_shape * index) + 5, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        for i in range(2):
            for j in range(3):
                if i == 0:
                    image = cv2.rectangle(image, (input_shape * j, 0), (input_shape * (j + 1), input_shape), (0, 255, 255), 2)
                else:
                    image = cv2.rectangle(image, (input_shape * j, 384), (input_shape * (j + 1), 1024), (0, 255, 255), 2)
        if index < 2:
            image = cv2.rectangle(image, (input_shape * index, 0), (((input_shape * index) + (w*2)), (h*2)), (0, 0, 255), -1)
            image = cv2.putText(image, label, ((input_shape * index) + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            image = cv2.rectangle(image, (input_shape * index, 440), (((input_shape * index) + (w*2)), (440 + (h * 2))), (0, 0, 255), -1)
            image = cv2.putText(image, label, ((input_shape * index) + 5, 399), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_inf_rectangle(image, full_boxes):
    
    for box in full_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf, cls = box[4], box[5]

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image

with dai.Device(create_pipeline()) as device:

    detIn = device.getInputQueue("detection_in")
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
    files = os.listdir('water')
    value = int(3 * ((input_shape/8) ** 2 + (input_shape/16) ** 2 + (input_shape/32) ** 2))
    
    for file in files:
        full_boxes = None
        img = cv2.imread(os.path.join('water', file))
        crop = split_images(img)

        total_time = time.time()

        for index, image in enumerate(crop):

            subwindow_time = time.time()
            image = image.transpose(2, 0, 1)
            lic_frame = dai.ImgFrame()
            lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
            lic_frame.setData(image)
            lic_frame.setWidth(input_shape)
            lic_frame.setHeight(input_shape)
            detIn.send(lic_frame)

            in_nn = qDet.get()
            layers = in_nn.getAllLayers()
            output = np.array(in_nn.getLayerFp16("output"))
            
            subwindow_inf_time = time.time() - subwindow_time
            subwindow_inf_time = round(subwindow_inf_time, 2)

            cols = output.shape[0] // value
            output = np.reshape(output, (value, cols))
            output = np.expand_dims(output, axis=0)

            total_classes = cols - 5

            boxes = non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            boxes = np.array(boxes[0])

            label = f"subwindow_inf_time: {subwindow_inf_time}"
            (w, h) = get_textbox_size(label)
            draw_inf_time(img, w, h, label, index)

            if boxes.size > 1:
                if img.shape == (1080, 1920, 3):
                    if index > 2:
                        boxes[:, 1] += 440
                        boxes[:, 3] += 440
                    boxes[:, 0] += (input_shape * (index % 3))
                    boxes[:, 2] += (input_shape * (index % 3))
                else :
                    if index > 1:
                        boxes[:, 1] += 384
                        boxes[:, 3] += 384
                    boxes[:, 0] += (input_shape * (index % 2))
                    boxes[:, 2] += (input_shape * (index % 2))

                if full_boxes is None:
                    full_boxes = boxes
                else:
                    full_boxes = np.vstack((full_boxes, boxes))
            
        full_inf_time = time.time() - total_time


        full_inf_time = round(full_inf_time, 2)
        label = f"full_window_inf_time: {full_inf_time}" 

        (w, h) = get_textbox_size(label)

        if img.shape == (1080, 1920, 3):
            img = cv2.rectangle(img, (1920-w, 1080-h), (1920, 1080), (0, 0, 255), -1)
            img = cv2.putText(img, (label), (1920-w, 1080-h+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else :
            img = cv2.rectangle(img, (1280-w, 1024-h), (1280, 1024), (0, 0, 255), -1)
            img = cv2.putText(img, (label), (1280-w, 1024-h+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img = draw_inf_rectangle(img, full_boxes)
        

        cv2.imwrite("result_img_"+ '_' + file[:-4] +'.jpg', img)
    
        
    