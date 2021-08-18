from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time

def create_pipeline():
    # pipeline 생성
    pipeline = dai.Pipeline()

    # yolo 탐지 노드 생성 및 설정
    nn = pipeline.createYoloDetectionNetwork()
    nn.setBlobPath('../models/custom_car.blob')

    nn.setConfidenceThreshold(0.5)
    nn.setNumClasses(2)
    nn.setCoordinateSize(4)
    nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    nn.setIouThreshold(0.5)

    # 이미지를 oak-d로 전송하기 위한 XLinkIn 노드 생성
    detection_in = pipeline.createXLinkIn()
    detection_in.setStreamName("detection_in")
    # XLinkIn 노드의 output을 yolo 노드의 input과 연결
    detection_in.out.link(nn.input)

    # 탐지 결과를 받을 수 있게 해주는 XLinkOut 노드 생성
    nnOut = pipeline.createXLinkOut()
    nnOut.setStreamName("nn")
    # yolo노드의 output을 XLinkOut 노드의 input과 연결
    nn.out.link(nnOut.input)

    return pipeline

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

files = ['../img/' + 'img_'+str(i)+'.jpg' for i in range(1, 11)]
labelMap = ["car", "pedestrian"]

with dai.Device(create_pipeline()) as device:

    # 생성한 XLinkIn 노드와 yolo 노드 큐 호출
    detIn = device.getInputQueue("detection_in")

    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)


    detections = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    for i in range (3):
        img = cv2.imread(files[i])
        lic_frame = dai.ImgFrame()
        lic_frame.setData(to_planar(img, (416, 416)))
        lic_frame.setType(dai.RawImgFrame.Type.YUV420p)
        lic_frame.setWidth(416)
        lic_frame.setHeight(416)
        detIn.send(lic_frame)

        detections = qDet.get().detections

        for detection in detections:
            bbox = frameNorm(img, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(img, labelMap[detection.label] + str(detection.confidence * 100), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.imwrite("result_img_" + str(i) + '.jpg', img)