#!/bin/bash

pip install ultralytics

# Run yolo training
yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=8
