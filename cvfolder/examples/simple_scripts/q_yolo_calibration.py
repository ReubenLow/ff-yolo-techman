import time
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import numpy as np
import requests
from io import BytesIO
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
from scipy.spatial.transform import Rotation as R
import json
import os
import sys

# -----------------------------------------------------------------------------
# Initialization and Configuration
# -----------------------------------------------------------------------------
# Initialize the YOLO model with a pre-trained model (YOLOv8n) and configure
# the Intel RealSense camera pipeline to capture both color and depth data.
# This setup is essential for the subsequent object detection and coordinate
# transformation processes.
# -----------------------------------------------------------------------------

# INIT YOLO
model = YOLO('yolov8n.pt')  # Ensure this is the correct path to your YOLO model

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)  # Enable color stream with specific resolution and format
cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)   # Enable depth stream with specific resolution and format

# Start the pipeline
profile = pipe.start(cfg)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)


# -----------------------------------------------------------------------------
# Function: capture_image
# -----------------------------------------------------------------------------
# Purpose:
# This function captures an image from the Intel RealSense camera's color stream
# and saves it as a JPEG file. The saved image is later used for object detection.
#
# Parameters:
# - color_frame: The color frame captured from the RealSense camera.
#
# Returns:
# - file_path: The path where the captured image is saved.
# -----------------------------------------------------------------------------

def capture_image(color_frame):
    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())
    
    # Save the captured frame to a specified file path
    file_path = "image.jpg"
    cv2.imwrite(file_path, color_image)
    print("Image captured and saved successfully!")

    return file_path


# -----------------------------------------------------------------------------
# Function: detect_objects_in_image
# -----------------------------------------------------------------------------
# Purpose:
# This function uses the YOLO model to detect objects in a captured image.
# It identifies the target object specified by the user and returns its bounding
# box coordinates.
#
# Parameters:
# - image_path: The path to the image file to be processed.
# - target_object: The object to detect in the image (e.g., "teddy bear").
#
# Returns:
# - A dictionary containing the bounding box coordinates (x1, y1, x2, y2) if the
#   target object is detected; otherwise, returns None.
# -----------------------------------------------------------------------------

def detect_objects_in_image(image_path, target_object):
    image = cv2.imread(image_path)  # Load the image
    results = model(image)  # Perform object detection using YOLO
    
    # Loop through detection results and find the target object
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            label = model.names[class_id]  # Get the label of the object

            if label == target_object:  # Check if this is the target object, e.g., teddy bear
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    return None  # If the target object is not detected, return None


# -----------------------------------------------------------------------------
# Function: get_coordinates
# -----------------------------------------------------------------------------
# Purpose:
# This function captures color and depth frames from the RealSense camera, detects
# the target object using YOLO, and calculates its real-world coordinates based on
# the depth data. It then saves these coordinates to a JSON file.
#
# Parameters:
# - pipe: The RealSense camera pipeline.
# - align: The alignment object used to align depth frames to color frames.
# - target_object: The object to detect and locate (default is "teddy bear").
#
# Returns:
# - A tuple containing the real-world coordinates (Z, X, Y) of the detected object
#   in meters, or None if the process fails.
#
# Simulated Camera View:
# The following diagram simulates the camera's view with the top-left corner as the origin (0, 0).
# The detected object is represented with a bounding box, and the center of the box (detected object)
# is marked with an asterisk (*) which corresponds to the center (x, y) in the code.

# (0,0) -----------------------------------------------> x-axis
#   |      ______________________________________
#   |     |                                      |
#   |     |                                      |
#   |     |                                      |
#   |     |       +---------------------+        |
#   |     |       |                     |        |
#   |     |       |      [Object]       |        |
#   |     |       |         *           |        |
#   |     |       |                     |        |
#   |     |       +---------------------+        |
#   |     |                                      |
#   |     |                                      |
#   |     |                                      |
#   |     |______________________________________|
#   v
# y-axis
#
# Legend:
# - [Object]: Detected object (e.g., teddy bear).
# - *: Center of the bounding box where the object's coordinates (x, y) are calculated.
# -----------------------------------------------------------------------------

def get_coordinates(pipe, align, target_object="teddy bear"):

    output_directory = "/home/reuben/techmanpy"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the target object name to a JSON file
    json_file_path = os.path.join(output_directory, "object.json")
    with open(json_file_path, "w") as json_file:
        json.dump(target_object, json_file, indent=4)

    print("get coordinates")
    
    # Capture frames from the RealSense camera
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames.")
        return None
    
    # Get intrinsic parameters of the RealSense camera
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data()) 

    # Save the captured color image
    image_path = capture_image(color_frame)

    # Perform object detection using YOLO
    gpt_output = detect_objects_in_image(image_path, target_object)
    if not gpt_output:
        print("Object detection failed.")
        return None

    print(gpt_output)

    # Calculate the center of the detected bounding box
    x, y = (gpt_output['x1'] + gpt_output['x2']) // 2, (gpt_output['y1'] + gpt_output['y2']) // 2
    x1, y1, x2, y2 = gpt_output['x1'], gpt_output['y1'], gpt_output['x2'], gpt_output['y2']

    # Draw bounding box and center on the image
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Thickness of the bounding box
    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(color_image, (x, y), thickness, color, -1)
    cv2.imshow('Image with Bounding Box', color_image)
    cv2.waitKey(1)  # Update the image window continuously

    # Get depth at the specified coordinates
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        print("Invalid depth data at the given coordinates.")
        return None
    
    # Print the distance to the detected object
    print(f"Distance to object: {depth}")

    # Calculate real-world coordinates using intrinsic parameters
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_frame.profile.as_video_stream_profile().intrinsics, [x, y], depth)
    
    print(f"x, y, z coordinates in m: {depth_point[0]}, {depth_point[1]}, {depth_point[2]}")

    # Convert coordinates to millimeters
    x_mm = depth_point[0] * 1000
    y_mm = depth_point[1] * 1000
    z_mm = depth_point[2] * 1000
    print(f"x, y, z coordinates in mm: {x_mm}, {y_mm}, {z_mm}")

    # Save the camera coordinates to a JSON file
    coordinates = {
        "X": depth_point[0],
        "Y": depth_point[1],
        "Z": depth_point[2]
    }
    with open("coordinates.json", "w") as json_file:
        json.dump(coordinates, json_file, indent=4)

    # Transform the coordinates to the robot's base frame
    get_transformed_coords(x_mm, y_mm, z_mm)

    return (depth_point[2], depth_point[0], depth_point[1])


# -----------------------------------------------------------------------------
# Function: get_transformed_coords
# -----------------------------------------------------------------------------
# Purpose:
# This function transforms the detected object's camera coordinates to the robot's
# base frame using a homogeneous transformation matrix. The transformation matrix
# is derived from known camera and robot coordinates, allowing the system to map
# camera detections to the robot's coordinate system.
#
# Parameters:
# - x, y, z: The real-world coordinates of the object in millimeters, as detected by the camera.
#
# Mathematical Overview:
# 1) **Camera Coordinate Calculation**:
#    The RealSense camera uses intrinsic parameters to deproject a 2D pixel (u, v) 
#    into 3D coordinates (X_c, Y_c, Z_c) in the camera's coordinate system:
#
#    ```
#    X_c = (u - cx) * Z_c / fx
#    Y_c = (v - cy) * Z_c / fy
#    Z_c = depth
#    ```
#
#    Where:
#    - (u, v): Pixel coordinates in the image.
#    - (cx, cy): Principal point (optical center) of the camera.
#    - (fx, fy): Focal lengths in the x and y axes.
#    - Z_c: The depth value at the pixel (u, v).
#
# 2) **Transformation to Robot Base Frame**:
#    The 3D coordinates in the camera's coordinate system (X_c, Y_c, Z_c) are 
#    transformed into the robot's base frame (X_r, Y_r, Z_r) using the homogeneous 
#    transformation matrix `T`:
#
#    ```
#    [X_r]     [R11 R12 R13 Tx]   [X_c]
#    [Y_r]  =  [R21 R22 R23 Ty] * [Y_c]
#    [Z_r]     [R31 R32 R33 Tz]   [Z_c]
#    [ 1 ]     [  0   0   0   1]   [ 1 ]
#    ```
#
#    Where:
#    - [X_r, Y_r, Z_r, 1]^T: The homogeneous coordinates in the robot's base frame.
#    - [X_c, Y_c, Z_c, 1]^T: The homogeneous coordinates in the camera's coordinate frame.
#    - [R]: 3x3 Rotation matrix derived from SVD (singular value decomposition).
#    - [T]: Translation vector (Tx, Ty, Tz) that aligns the camera frame with the robot base frame.
#
# Returns:
# - None. The transformed coordinates are saved to a JSON file for further use.
# -----------------------------------------------------------------------------


def get_transformed_coords(x, y, z):

    # Known points in the camera coordinate system (X_c, Y_c, Z_c) and corresponding robot base coordinates (X_r, Y_r, Z_r)
    camera_points = np.array([
        [188.7, 12.85, 706.00], #1
        [-43.55, 11.63, 782.00], #2
        [-140.57, 19.35, 689.00], #3
        [-142.61, 27.53, 476.00], #4
        [-50.19, 22.35, 587.99], #5
        [124.88, 26.59, 435.00], #6
        [-35.85, 29.84, 420.00], #7
        [96.11, 20.19, 582.00] #8
    ])

    robot_points = np.array([
        [-234.22, -637.53, 65.18], #1
        [-37.2, -707.2, 73.02], #2
        [59.27, -633.31, 76.42], #3
        [69, -425.86, 73.67], #4
        [-20.84, -525, 73.61], #5
        [-180.38, -388.39, 66.38], #6
        [-25.89, -372.82, 71.37], #7
        [-167.93, -527.66, 68.16] #8
    ])

    # Center the points (subtract the mean)
    camera_mean = np.mean(camera_points, axis=0)
    robot_mean = np.mean(robot_points, axis=0)

    camera_points_centered = camera_points - camera_mean
    robot_points_centered = robot_points - robot_mean

    # Compute the covariance matrix
    H = np.dot(camera_points_centered.T, robot_points_centered)

    # Singular Value Decomposition (SVD) to compute the rotation matrix
    U, S, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) should be +1)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t_vector = robot_mean - np.dot(R_matrix, camera_mean)

    print("Estimated Rotation Matrix R:")
    print(R_matrix)
    print("Estimated Translation Vector t:")
    print(t_vector)

    # Construct the 4x4 homogeneous transformation matrix T
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = t_vector

    print("Homogeneous Transformation Matrix T:")
    print(T)

    # Example camera point in homogeneous coordinates (X_c, Y_c, Z_c, 1)
    example_camera_point = np.array([x, y, z, 1])

    # Transform the point to the robot's base frame
    transformed_point = T @ example_camera_point

    # Extract the transformed coordinates
    X_r_new, Y_r_new, Z_r_new = transformed_point[0], transformed_point[1], transformed_point[2]

    # Create a dictionary of the transformed coordinates
    transformed = {
        "X": X_r_new,
        "Y": Y_r_new,
        "Z": Z_r_new,
        "Rx": 90.04,   # fixed rotation values, to be edited according to the wanted rotation
        "Ry": -0.68,
        "Rz": 12.96
    }

    # Save the transformed coordinates to a JSON file
    output_directory = "/home/reuben/techmanpy"
    os.makedirs(output_directory, exist_ok=True)
    json_file_path = os.path.join(output_directory, "coordinates.json")
    with open(json_file_path, "w") as json_file:
        json.dump(transformed, json_file, indent=4)

    print(f"Transformed coordinates in robot's base frame: X={X_r_new}, Y={Y_r_new}, Z={Z_r_new}")


# -----------------------------------------------------------------------------
# Function: SpeechListener
# -----------------------------------------------------------------------------
# Purpose:
# This function listens for voice commands using the microphone and recognizes
# speech to identify the target object. The recognized object name is then used
# for subsequent detection and coordinate transformation.
#
# Returns:
# - The detected object name if successful, or None if no valid command is detected.
# -----------------------------------------------------------------------------

def SpeechListener():
    keywords = ["cup", "teddy bear", "orange", "sports ball", "vase"]

    # Initialize the speech recognizer
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Adjust energy threshold for ambient noise
            print("Adjusting threshold...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Ready to listen...")

            print("Listening for 5 seconds...")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing Speech....")
            prompt = r.recognize_google(audio)
            result = prompt.lower()
            
            print("Recognized speech: ", result)

            # Check if any keyword is in the recognized speech
            for keyword in keywords:
                if keyword in result:
                    detected_keyword = keyword
                    print(f"'{keyword}' detected in request.")
                    if result == "this" or result == "vas" or result == "pause" or result == "false" or result == "voss": # Handle mispronunciations
                        detected_keyword = "vase"
                    return detected_keyword
            
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nTerminating...")
        sys.exit(0)


# -----------------------------------------------------------------------------
# Function: main
# -----------------------------------------------------------------------------
# Purpose:
# This is the main function that controls the overall flow of the program. It
# prompts the user to choose between typing the object name or using voice
# recognition for detection. The detected object is then processed to obtain
# its coordinates, which are transformed and saved.
#
# Parameters:
# - None
#
# Returns:
# - None
# -----------------------------------------------------------------------------

def main():
    print("Starting object detection...")

    while True:
        # Prompt the user to either type the object name or use voice recognition
        choice = input("Enter 't' to type requested object or 'l' to say requested object or type 'q' to exit: ").strip()

        if choice.lower() == "q":
            print("Exiting...")
            break
        elif choice.lower() == "t":
            target_object = input("Please enter the object you want to detect (e.g., 'teddy bear'):").strip()
            if not target_object:
                print("Invalid input. Please enter a valid object name.")
                continue  # Ask again if the input is invalid

            detection_count = 0  # Reset detection count

            while detection_count < 2:  # Keep looking for the object until it's detected twice
                try:
                    coords = get_coordinates(pipe, align, target_object=target_object)
                    if coords:
                        x, y, z = coords
                        detection_count += 1  # Increment detection count
                        print(f"Detection {detection_count}: Coordinates - X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

                        if detection_count < 2:
                            print("Waiting for the next detection...")
                            continue  # Continue looking for the second detection
                        else:
                            break  # Exit the loop when detected twice

                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Retrying to get valid coordinates...")

            if detection_count >= 2:
                print("Object detected twice successfully.")
            else:
                print("Failed to detect the object twice. Restarting the detection process.")

        elif choice.lower() == "l":
            target_object = SpeechListener()

            # Ensure the target_object is not None
            if target_object is None:
                print("No object detected from speech. Please try again.")
                continue

            detection_count = 0  # Reset detection count

            while detection_count < 2:  # Keep looking for the object until it's detected twice
                try:
                    coords = get_coordinates(pipe, align, target_object=target_object)
                    if coords:
                        x, y, z = coords
                        detection_count += 1  # Increment detection count
                        print(f"Detection {detection_count}: Coordinates - X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

                        if detection_count < 2:
                            print("Waiting for the next detection...")
                            continue  # Continue looking for the second detection
                        else:
                            break  # Exit the loop when detected twice

                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Retrying to get valid coordinates...")

            if detection_count >= 2:
                print("Object detected twice successfully.")
            else:
                print("Failed to detect the object twice. Restarting the detection process.")

    # Stop the pipeline when finished
    pipe.stop()


if __name__ == "__main__":
    main()
