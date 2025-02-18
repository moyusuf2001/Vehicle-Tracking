# Vehicle-Tracking
This project implements object tracking in videos using OpenCV and YOLO-based object detection. It processes video frames, detects objects, assigns unique IDs for each object, and tracks their movement across frames. The output is saved as a new video file with bounding boxes and tracking information.

Object Detection with YOLOv4
This repository implements object detection using OpenCV’s DNN module and the YOLOv4 model. It detects objects in images or video frames, visualizes the results with bounding boxes, and provides configurable thresholds for detection and non-maximum suppression.

Features
Object detection with YOLOv4 using OpenCV DNN.
GPU acceleration (CUDA-enabled) for faster processing.
Configurable detection confidence and non-maximum suppression thresholds.
Visualization of bounding boxes and object labels.
Requirements
Python 3.x
OpenCV (with DNN module support)
Numpy
CUDA-enabled GPU (optional, for faster processing)

Install the required libraries:

"pip install opencv-python opencv-contrib-python numpy"

Configuration
This repository includes the necessary YOLOv4 weights, configuration, and class names within the project. Ensure the paths are set correctly for loading them.

Customization
You can adjust the following parameters:

Confidence Threshold: Minimum confidence for detecting an object (default: 0.5).
NMS Threshold: Non-maximum suppression threshold to avoid multiple detections for the same object (default: 0.4).
Image Size: The input image size for the model (default: 608).

File Structure

/dnn_model
    /yolov4.weights
    /yolov4.cfg
    /classes.txt
/object_detection.py
