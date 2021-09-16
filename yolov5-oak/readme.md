# YOLOv5s-OAK
## 탑재과정
```flow
start=>start: Custom Data
end=>end
o1=>operation: train.py(python train.py --img 416 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt --device 0)
cond1=>condition: best.pth
o2=>operation: export.py(python export.py --weights runs/train/exp10/weights/best.pt --img 416 --batch 1 --device cpu --include "onnx" --simplify)
cond2=>conditiion: best.onnx
o3=>operation: model optimizer(Intel OpenVino)(python3 mo.py --input_model ~/yolov5/best.onnx --model_name yolov5s_custom --data_type FP16 --output_dir ~/yolvo5/ --input_shape "[1, 3, 416, 416]" --reverse_input_channel --scale 255
)
cond3=>condition: best.xml, best.bin
o4=>operation: compile tool(Intel OpenVino)(./compile_tool -m ~/yolvo5/yolov5s_custom.xml -ip U8 -d MYRIAD -VPU_NUMBER_OF_SHAVES 6)
cond4=>condition: best.blob
o5=>operation: gen2_yolov5_inf.py
start->o1->cond1->o2->cond2->o3->cond3->o4->cond4->o5
```