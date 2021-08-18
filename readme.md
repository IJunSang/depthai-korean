# Ubuntu 18.04 LTS 기준
## install dependencies
```shell
sudo wget -qO- http://docs.luxonis.com/_static/install_dependencies.sh | bash
```
## VMware 사용 시
- Virtual Machine settings -> USB controller -> USB compatibility를 usb 3.1 또는 3.0으로 설정
- Intel Movidius MyriadX, Intel VSC Loopback Device 또는 Intel Luxonis Device를 가상머신으로 연결해 주어야 함
- 아래 쉘 스크립트를 실행하여 USB 권한 할당
```shell
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## VirtualBox 사용 시
- Oracle VM VirtualBox Extension Pack 설치
- Devices -> USB Settings 에서 USB3 (xHCI) 활성화
- Devices -> USB에서 Intel Movidius MyriadX, Intel VSC Loopback Device 또는 Intel Luxonis Device 체크

## 이외
- https://docs.luxonis.com/projects/api/en/latest/install/#supported-platforms 링크 참조

## 패키지 설치
```shell
python3 -m pip install depthai
```

# 테스트
## cloning depthai-python repository
```
git clone https://github.com/luxonis/depthai-python.git
cd depthai-python
```
## example 레포지토리를 위한 요구사항 설치
```
cd examples
python3 install_requirements.py
```
## rgb_preview.py를 실행하여 작동 확인
```
python3 rgb_preview.py
```

# Source
## inference.py
- https://aihub.or.kr/ 의 도로주행영상 자료를 이용하여 자동차 탐지 학습
- img 폴더 안의 이미지를 읽어 추론 결과 result_img_*.jpg의 형식으로 저장

## aqua_yolo.py
- roboflow의 public dataset중 하나인 aquarium dataset(https://public.roboflow.com/object-detection/aquarium)을 이용하여 학습

## chess_yolo.py
- https://blog.roboflow.com/deploy-luxonis-oak/ 링크 참조하여 제작