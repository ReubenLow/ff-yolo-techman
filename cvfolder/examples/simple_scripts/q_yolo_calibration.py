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

# INIT YOLO
model = YOLO('yolov8n.pt')  # Ensure this is the correct path to your YOLO model

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)
cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)

# Start the pipeline
profile = pipe.start(cfg)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)


#YOLO
def capture_image(color_frame):
    # Convert the color frame to a numpy array if it's not already
    color_image = np.asanyarray(color_frame.get_data())
    
    # Save the captured frame to the specified file path
    file_path = "image.jpg"
    cv2.imwrite(file_path, color_image)
    print("Image captured and saved successfully!")

    return file_path


# Latest
def detect_objects_in_image(image_path, target_object):
    image = cv2.imread(image_path)  # Load the image
    results = model(image)  # Perform object detection using YOLO
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            label = model.names[class_id]  # Get the label of the object

            if label == target_object:  # Check if this is the target object, e.g. teddy bear
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    return None  # If the target object is not detected, return None


# Latest
def get_coordinates(pipe, align, target_object="teddy bear"):

    # For reference
    # dpt_frame = pipe.wait_for_frames().get_depth_frame().as_depth_frame()
    # pixel_distance_in_meters = dpt_frame.get_distance(x,y)

    output_directory = "/home/reuben/techmanpy"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the file path for JSON file
    json_file_path = os.path.join(output_directory, "object.json")

    with open(json_file_path, "w") as json_file:
        json.dump(target_object, json_file, indent=4)

    print("get coordinates")
    
    # Capture frames
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    # depth_frame = aligned_frames.get_depth_frame()
    depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames.")
        return None
    
    # Get intrinsic parameters of realsense
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data()) 

    # Save the captured color image
    image_path = capture_image(color_frame)

    # Object detection using YOLO
    gpt_output = detect_objects_in_image(image_path, target_object)
    if not gpt_output:
        print("Object detection failed.")
        return None

    print(gpt_output)

    # Extract the coordinates from YOLO output
    x, y = (gpt_output['x1'] + gpt_output['x2']) // 2, (gpt_output['y1'] + gpt_output['y2']) // 2
    x1, y1, x2, y2 = gpt_output['x1'], gpt_output['y1'], gpt_output['x2'], gpt_output['y2']

    # Draw bounding box and center on the image
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2  # Thickness of the bounding box
    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(color_image, (x, y), thickness, color, -1)
    cv2.imshow('Image with Bounding Box', color_image)
    cv2.waitKey(1)  # Update the image window continuously

    # Get depth at the specified coordinates (distance from the camera to a point in the scene along the camera's view direction.)
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        print("Invalid depth data at the given coordinates.")
        return None
    
    # Distance to object
    print(f"Distance to object: {depth}")

    # Calculate real-world coordinates using intrinsic parameters
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_frame.profile.as_video_stream_profile().intrinsics, [x, y], depth)
    
    # print(f"Real-world coordinates: {depth_point[2]}, {depth_point[0]}, {depth_point[1]}")
    print(f"x, y, z coordinates in m: {depth_point[0]}, {depth_point[1]}, {depth_point[2]}")
    # print(f"Z-axis value: {depth_point[2]}")

    # Save camera coords to json
    x, y, z = depth_point[0], depth_point[1], depth_point[2]

    # Convert to mm
    x_mm = x*1000
    y_mm = y*1000
    z_mm = z*1000
    print(f"x, y, z coordinates in mm: {x_mm}, {y_mm}, {z_mm}")


    coordinates = {
        "X": x,
        "Y": y,
        "Z": z
    }
    with open("coordinates.json", "w") as json_file:
        json.dump(coordinates, json_file, indent=4)

    get_transformed_coords(x_mm, y_mm, z_mm)

    return (depth_point[2], depth_point[0], depth_point[1])

def get_transformed_coords(x, y, z):

    # Points in the camera coordinate system (X_c, Y_c, Z_c)
    camera_points = np.array([
        # [67.17, 27.76, 442.00],
        # [-227.05, 26.86, 560.5],
        # [-53.06, 22.75, 598.5],
        [188.7, 12.85, 706.00], #1
        [-43.55, 11.63, 782.00], #2
        [-140.57, 19.35, 689.00], #3
        [-142.61, 27.53, 476.00], #4
        [-50.19, 22.35, 587.99], #5
        [124.88, 26.59, 435.00], #6
        [-35.85, 29.84, 420.00], #7
        [96.11, 20.19, 582.00] #8
    ])

    # Corresponding points in the robot's base coordinate system (X_r, Y_r, Z_r)
    robot_points = np.array([
        # [-90.91, -401.30, 72.57],
        # [122.17, -514.19, 67.10],
        # [8.54, -547.79, 71.40],
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

    # Singular Value Decomposition (SVD)
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

    # Apply sign correction if necessary
    # Assuming you find x and y need to be inverted
    X_r_new, Y_r_new, Z_r_new = transformed_point[0], transformed_point[1], transformed_point[2]

    transformed = {
        "X": X_r_new,
        "Y": Y_r_new,
        "Z": Z_r_new,
        "Rx": 90.04,
        "Ry": -0.68,
        "Rz": 12.96
    }

    output_directory = "/home/reuben/techmanpy"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the file path for JSON file
    json_file_path = os.path.join(output_directory, "coordinates.json")

    with open(json_file_path, "w") as json_file:
        json.dump(transformed, json_file, indent=4)

    print(f"Transformed coordinates in robot's base frame: X={X_r_new}, Y={Y_r_new}, Z={Z_r_new}")


# Example usage:
# pipe and align should be initialized RealSense pipeline and align objects
# get_coordinates(pipe, align)
        # results = model(color_image)
        # print(results)
        # for result in results:
            # boxes = result.boxes
            # for box in boxes:
                # center_x, center_y = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                # confidence = box.conf[0]
                # class_id = int(box.cls[0])
                # label = model.names[class_id]

                # # Check if the detected object's confidence is above a threshold
                # if label == targetObject and confidence > 0.2:
                    # # Get the depth at the center of the bounding box
                    # depth = depth_frame.get_distance(center_x, center_y)
                    # if depth > 0:
                        # # Convert depth to real world coordinates
                        # depth_point = rs.rs2_deproject_pixel_to_point(
                            # depth_frame.profile.as_video_stream_profile().intrinsics, [center_x, center_y], depth)
                        #print(depth_point[2], depth_point[0], depth_point[1])
                        #return (depth_point[2], depth_point[0], depth_point[1])  # Return as (x, y, z)
        
        # Check if the timeout of 5 seconds has been reached
        #if time.time() - start_time > 15:
        #    break

def SpeechListener():
    keywords = ["cup", "teddy bear", "orange", "sports ball", "vase"]

    # Initialize recognizer
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Adjust energy threshold
            print("Adjusting threshold...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Ready to listen...")

            print("Listening for 5 seconds...")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing Speech....")
            prompt = r.recognize_google(audio)
            result = prompt.lower()
            
            print("Recognized speech: ", result)

            # Find requested object
            # Check if any keyword is in the recognized speech
            for keyword in keywords:
                if keyword in result:
                    detected_keyword = keyword
                    print(f"'{keyword}' detected in request.")
                    if result == "this" or result == "vas" or result == "pause" or result == "false" or result == "voss": # In case it can't recognize the pronounciation of 'vase'
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


def main():
    print("Starting object detection...")

    while True:
        # Prompt user for the object to detect
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

            while detection_count < 2:  # Keep looking for the object until it's detected twice, for better lighting and exposure
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
