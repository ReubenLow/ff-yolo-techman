import time
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import numpy as np
import requests
from io import BytesIO
import base64
import spacy
from openai import OpenAI
import argparse
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time
import json
import keyboard
import speech_recognition as sr
import pyttsx3
import numpy as np
import pyrealsense2 as rs
from PIL import Image

# Initialize YOLO model
model = YOLO('yolov8n.pt')  

# Function to detect objects in image.jpg
def detect_objects_in_image(image_path):
    image = cv2.imread(image_path)  # Load the image
    results = model(image)  # Perform object detection using YOLO
    
    for result in results:
        boxes = result.boxes
        if boxes:
            box = boxes[0]  # Consider the first detected object
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    return None  # If no objects are detected, return None

def get_coordinates(pipe, align):
    print("get coordinates")
    
    # Load the image instead of capturing it
    image_path = "image.jpg"
    gpt_output = detect_objects_in_image(image_path)
    
    if not gpt_output:
        print("Object detection failed or no objects detected.")
        return None

    print(gpt_output)
    
    # Load the image for display
    color_image = cv2.imread(image_path)
    
    # Extract the coordinates from YOLO output
    x, y = (gpt_output['x1'] + gpt_output['x2']) // 2, (gpt_output['y1'] + gpt_output['y2']) // 2
    x1, y1, x2, y2 = gpt_output['x1'], gpt_output['y1'], gpt_output['x2'], gpt_output['y2']
    
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Thickness of the bounding box
    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(color_image, (x, y), thickness, color, -1)
    cv2.imshow('Image with Bounding Box', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate real-world coordinates using the depth from the previously captured frame
    depth_frame = pipe.wait_for_frames().get_depth_frame()
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        print("Invalid depth data at the given coordinates.")
        return None

    depth_point = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, [x, y], depth)
    
    print(f"Real-world coordinates: {depth_point[2]}, {depth_point[0]}, {depth_point[1]}")
    return (depth_point[2], depth_point[0], depth_point[1])

def main():
    # Initialize RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)
    cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)

    # Start the pipeline
    profile = pipe.start(cfg)

    # Align depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Use coordinates from image.jpg
    print("before coords")
    valid_coords = False
    while not valid_coords:
        try:
            coords = get_coordinates(pipe, align)
            if coords:
                x, y, z = coords
                print(f"Coordinates to move to: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
                valid_coords = True
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying to get valid coordinates...")

    # Stop the pipeline
    pipe.stop()

if __name__ == "__main__":
    main()
