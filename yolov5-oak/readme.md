# YOLOv5s-OAK

## 필요 환경
---
python, pytorch(설치는 (https://pytorch.org/get-started/locally/) 참조), depthai (설치는 (https://github.com/IJunSang/depthai-korean) 참조), Intel OpenVINO (아래 참조), YOLOv5 Repo (아래 참조)

## 탑재과정
---
![flowchart](./flowchart.png)

## YOLOv5 Repository
---
[ultralytics/yolov5yolov5](https://github.com/ultralytics/yolov5)

## Intel OpenVINO
---
[Donlaod page](https://software.seek.intel.com/openvino-toolkit)

## 작업 과정
---
- custom data를 준비한다 (양식은 YOLOv5 Repo 참조, 이미지폴더와 라벨폴더, 클래스 이름이 들어있는 .names 파일, train이미지 경로를 넣은 txt파일, test용 이미지 경로를 넣은 txt 파일)

- dataset.yaml을 생성한다 (path를 dataset경로로, train, val, test에 각각 이미지 파일 경로가 들어가있는 텍스트파일, nc 수 클래스 수에 맞게 작성, names 는 라벨이름으로 작성, 레포 안 dataset.yaml 참조)

- flowchart에서 실행되는 명령어들을 참조하여 학습, 추출 진행

## 설치 완료 후
### openvino compile tool 사용시 libinference_engine.so cannot open shared object ~ 문제 생길경우

```
source <INSTALL_DIR>/bin/setupvars.sh -pyver (your_pyversion)
```
## Custom Data
---
[AIhub 도로주행영상](https://aihub.or.kr/aidata/8007) 중 train/valid -> 도심로 야간일몰_맑음_30_전방, 도심로 주간일출_강우_30_전방, 자동차전용도로_야간일몰_맑음 30_전방, 자동차전용도로 주간일출_안개_30_전방 사용

## YOLOv5 라벨 format
---
Darknet TXT 라벨 format을 따름 (object-class, x, y, width, height)
object-class: integer number of object from 0 to (classes-1)
x, y, width, height: float values relative to width and height of image, from (0.0 to 1.0)
x, y are the center of rectangle
ex) x = absolute_x / image_width

폴더 안에 각각 train, valid 시 들어갈 이미지들의 경로, 파일명이 포함된 txt파일이 있어야 함(custom_data 폴더 안 train.txt, valid.txt 참조)