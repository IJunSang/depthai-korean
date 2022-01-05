import cv2
import depthai as dai
from numpy.core.numeric import full
from torch._C import _replace_overloaded_method_decl
from util.functions import non_max_suppression
import numpy as np
import sys
import os
import time

labelMap = ["Person"]
IMG_SIZE = 640
HORIZONTAL_OVERLAP = 50

def create_pipeline():
    pipeline = dai.Pipeline()

    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath('water_cropped.blob')
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

def check_resolution(resolution):
    if resolution == (1080, 1920, 3):
        return 440
    elif resolution == (1024, 1280, 3):
        return 384
    else:
        return 80

def get_range(resolution):
    if resolution == (1080, 1920, 3):
        return 3
    else:
        return 2

def split_images(img):
    overlap = check_resolution(img.shape)
    loop_range = get_range(img.shape)
    crop = np.empty(((loop_range*2), IMG_SIZE, IMG_SIZE, 3))
    count = 0

    for i in range(2):
        for j in range(loop_range):
            left = (IMG_SIZE * j) - (HORIZONTAL_OVERLAP * j)
            right = (IMG_SIZE * (j + 1) - (HORIZONTAL_OVERLAP * j))
            if i == 0:
                crop[count] = img[0:IMG_SIZE, left:right, :]
            elif i == 1:
                crop[count] = img[overlap:(overlap+IMG_SIZE), left:right, :]

            count += 1

    return crop

def get_textbox_size(label):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    return (w, h)

def draw_inf_time(image, w, h, label, index):
    loop_range = get_range(image.shape)
    overlap = check_resolution(image.shape)
    for i in range(2):
        for j in range(loop_range):
            left_top = (IMG_SIZE * j) - (HORIZONTAL_OVERLAP * j)
            right_down = (IMG_SIZE * (j + 1)) - (HORIZONTAL_OVERLAP * j)
            if i == 0:
                image = cv2.rectangle(image, (left_top, 0), (right_down, IMG_SIZE), (0, 255, 255), 2)
            else:
                image = cv2.rectangle(image, (left_top, overlap), (right_down, (overlap + IMG_SIZE)), (0, 255, 255), 2)

    left_top = (IMG_SIZE * (index % loop_range)) - (HORIZONTAL_OVERLAP * (index % loop_range))
    if index < loop_range:
        height = 0
    else:
        height = overlap     
    image = cv2.rectangle(image, (left_top, height), ((left_top + (w*2)), (height + (h * 2))), (0, 0, 255), -1)
    image = cv2.putText(image, label, (left_top + 5, (height + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_inf_rectangle(image, full_boxes):
    
    for box in full_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf, cls = box[4], box[5]

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image

def check_iou(full_boxes):
    i = 0
    j = 0
    max_size = len(full_boxes)
    while i < max_size:
        min_x1, min_y1, max_x1, max_y1 = int(full_boxes[i][0]), int(full_boxes[i][1]), int(full_boxes[i][2]), int(full_boxes[i][3])
        size_box1 = (max_x1 - min_x1) * (max_y1 - min_y1)
        j = i + 1
        while j < max_size:
            min_x2, min_y2, max_x2, max_y2 = int(full_boxes[j][0]), int(full_boxes[j][1]), int(full_boxes[j][2]), int(full_boxes[j][3])
            size_box2 = (max_x2 - min_x2) * (max_y2 - min_y2)
            intersection_x_length = min(max_x1, max_x2) - max(min_x1, min_x2)
            intersection_y_length = min(max_y1, max_y2) - max(min_y1, min_y2)
            
            if intersection_x_length < 0 or intersection_y_length < 0: 
                j += 1
                continue

            intersection = intersection_x_length * intersection_y_length
            iou = intersection / ((size_box1 + size_box2) - intersection)

            if iou > 0:
                if intersection / size_box1 >= 0.7:
                    i += 1
                    min_x1, min_y1, max_x1, max_y1 = int(full_boxes[i][0]), int(full_boxes[i][1]), int(full_boxes[i][2]), int(full_boxes[i][3])
                    size_box1 = (max_x1 - min_x1) * (max_y1 - min_y1)
                    i -= 1
                    full_boxes = np.delete(full_boxes, i, 0)
                    max_size = len(full_boxes)
                    j = i + 1
                    continue
                elif intersection / size_box2 >= 0.7:
                    full_boxes = np.delete(full_boxes, j, 0)
                    max_size = len(full_boxes)
                    continue

            j += 1

        i += 1
    
    return full_boxes

with dai.Device(create_pipeline()) as device:

    detIn = device.getInputQueue("detection_in")
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
    files = os.listdir('water')
    value = int(3 * ((IMG_SIZE/8) ** 2 + (IMG_SIZE/16) ** 2 + (IMG_SIZE/32) ** 2))
    
    for file in files:
        full_boxes = None
        img = cv2.imread(os.path.join(os.getcwd(), 'water', file))
        crop = split_images(img)
        overlap = check_resolution(img.shape)
        loop_range = get_range(img.shape)

        total_time = time.time()

        for index, image in enumerate(crop):
            subwindow_time = time.time()
            image = image.transpose(2, 0, 1)
            lic_frame = dai.ImgFrame()
            lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
            lic_frame.setData(image)
            lic_frame.setWidth(IMG_SIZE)
            lic_frame.setHeight(IMG_SIZE)
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
                if index > (loop_range - 1):
                    boxes[:, 1] += overlap
                    boxes[:, 3] += overlap
                boxes[:, 0] += (IMG_SIZE * (index % loop_range) - (HORIZONTAL_OVERLAP * (index % loop_range)))
                boxes[:, 2] += (IMG_SIZE * (index % loop_range) - (HORIZONTAL_OVERLAP * (index % loop_range)))

                if full_boxes is None:
                    full_boxes = boxes
                else:
                    full_boxes = np.vstack((full_boxes, boxes))
            
        full_inf_time = time.time() - total_time


        full_inf_time = round(full_inf_time, 2)
        label = f"full_window_inf_time: {full_inf_time}" 

        (w, h) = get_textbox_size(label)

        img = cv2.rectangle(img, (img.shape[1] - w, img.shape[0] - h), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
        img = cv2.putText(img, (label), (img.shape[1]-w, img.shape[0]-h+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # np.set_printoptions(suppress=True)
        if full_boxes is not None:
            full_boxes = full_boxes.astype('int32')
            # print(full_boxes)            
            full_boxes = check_iou(full_boxes)
            img = draw_inf_rectangle(img, full_boxes)

        cv2.destroyAllWindows()

        cv2.imwrite("./cropped/result_img_cropped"+ '_' + file[:-4] +'.jpg', img)