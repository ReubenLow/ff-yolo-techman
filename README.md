# ff-yolo-techman

This project integrates YOLO object detection with robotic movement to pick up objects based on coordinates identified by the object detection system. 
The project is divided into two main folders: 
 - YOLO Object Detection: Detects objects in the camera's field of view and outputs their coordinates.
 - Robot Movement: Uses the detected coordinates to guide the robotic arm to pick up the specified object.

## Directory layout
```
├── cvfolder/
│   ├── examples/
│   │   ├── simple_scripts/
│   │   │   ├── q_yolo_calibration.py
│   │   │   ├── object.json
│   │   │   ├── coordinates.json
├── techmanfolder/
│   ├── testfile.py
└── README.md
```

## Installation
The installation steps for setting up the YOLO object detection and the Techman robotic arm control scripts are provided in their respective folders. Please refer to the README files within the cvfolder and techmanfolder for detailed installation instructions.
