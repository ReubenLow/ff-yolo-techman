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

## Usage
### Step 1: Run Object Detection

1. Navigate to the YOLO object detection folder:
``` bash
cd cvfolder/examples/simple_scripts
```
2. Run the object detection script to detect objects and output their coordinates:
``` bash
python q_yolo_calibration.py
```
- The detected object's name is saved in object.json.
- The coordinates are saved in coordinates.json.

### Note: You must run q_yolo_calibration.py first to obtain the coordinates of the object. These coordinates will be used by the robotic arm to pick up the object.

## Step 2: Run Robot Movement Script
1. Navigate to the Techman robot movement folder
``` bash
cd techmanfolder
```
2. Run the script to move the robot arm based on the detected coordinates:
``` bash
python3 testfile.py <robot-ip-address>
```
The robot will move to pick up the object specified in object.json using the coordinates from coordinates.json.

### Future Updates
Future updates will focus on integrating both the object detection and robot movement scripts into a single script. This will streamline the process, allowing the manipulator and camera to detect an object and move the robotic arm in one operation.

