#!/usr/bin/env python

import sys
import asyncio
import time
from enum import Enum
import techmanpy
from techmanpy import TechmanException
import copy
import json

# -----------------------------------------------------------------------------
# Initialization and Configuration
# -----------------------------------------------------------------------------
# This section handles the initial loading of data and configuration settings.
# It reads the robot's target coordinates and object information from JSON files,
# adjusts the coordinates based on the object type, and prints the final coordinates.
# This setup is essential for the subsequent robotic operations.
# It also assigns the robot's initial state and defines the state enumeration.
# and adds the Y and Z offsets for different objects.

# Specify the path to the JSON files, change these paths to your directory as needed
json_file_path = "/home/reuben/techmanpy/coordinates.json"
obj_file_path = "/home/reuben/techmanpy/object.json"

# Load the coordinates from the JSON file
with open(json_file_path, "r") as file:
   data = json.load(file)

# Load the specified object details from the JSON file
with open(obj_file_path, "r") as file:
   target_object = json.load(file)

# Extract the coordinate values and create an array (list)
itemcoor = list(data.values())

# Make a copy of the coordinates to adjust them for the movement
itemupcoor = copy.deepcopy(itemcoor)
itemupcoor[2] += 200  # Apply offset in z-axis
itemupcoor[0] -= 38   # Apply offset in x-axis
itemcoor[0] -= 38     # Apply offset in x-axis

# Apply y-offset for different objects due to imperfect calibration
# Adjusting based on the target object
if target_object == "orange":
   itemupcoor[1] -= 35  # Offset in y-axis for "orange"
   itemcoor[1] -= 35    # Offset in y-axis for "orange"
elif target_object == "vase":
   itemupcoor[1] -= 40  # Offset in y-axis for "vase"
   itemcoor[1] -= 40    # Offset in y-axis for "vase"
elif target_object == "cup":
   itemupcoor[1] -= 35  # Offset in y-axis for "cup"
   itemcoor[1] -= 35    # Offset in y-axis for "cup"
else:
   itemupcoor[1] -= 20  # Default offset in y-axis for other objects
   itemcoor[1] -= 20    # Default offset in y-axis for other objects

# Print the calculated coordinates for debugging purposes
print(itemcoor)
print(itemupcoor)

# -----------------------------------------------------------------------------
# State Enumeration Definition
# -----------------------------------------------------------------------------
# This block defines an enumeration for the robot's operational states.
# Enumerations are used for clarity and to prevent errors related to using 
# plain integers or strings. Each state represents a different phase of 
# the robot's task, such as moving to a coordinate, picking an object, or placing it.

class State(Enum):
   NEUTRAL = 0
   PICK = 1
   PLACE = 2
   COOR1 = 3
   COOR2 = 4
   COOR3 = 5
   COOR4 = 6
   COOR5 = 7
   COOR6 = 8
   START = 9
   HOME = 10
   END = 11
   # Return object to original position states
   PICK2 = 12
   PLACE2 = 13
   RETURN1 = 14
   RETURN2 = 15
   RETURN3 = 16
   RETURN4 = 17
   RETURN5 = 18
   RETURN6 = 19
   RETURN7 = 20

# Initialize the robot's state to 'HOME'
state = State.HOME

# -----------------------------------------------------------------------------
# Function: move
# -----------------------------------------------------------------------------
# Purpose: 
# This function controls the robot's movement to a specified set of coordinates.
# It transitions the robot through various states (e.g., moving to pick an object, 
# moving to place an object) and assigns queue tags for each movement to ensure 
# sequential execution.
#
# Parameters:
# - conn: The connection object to the robot, which allows sending movement commands.
# - arr: A list of coordinates to which the robot should move.
#
# Scenarios:
# This function is used whenever the robot needs to move to a new position. 
# The specific state of the robot determines which set of coordinates it moves to.
#
# Note: This function is an asynchronous function, allowing for non-blocking execution.
# Note: RETURNING THE OBJECT IS OPTIONAL AND IS JUST A DEMONSTRATION OF THE ROBOT'S CAPABILITIES.

async def move(conn, arr):
   global state
   tag = 1  # Initialize the queue tag
   
   # Transition between states and assign queue tags for each state
   if state == State.HOME:  # Home position
      # Move from Home to the position before picking the Cup
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H]*  UP      |
|                         |
|                         |
|                         |
|                         |
|          [Cup]          |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.COOR1
      tag = 1
   elif state == State.COOR1:  # Position before picking
      # move to above the cup
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H]           |
|                         |
|                         |
|                         |
|                         |
|          [Cup]* UP      |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.COOR2
      tag = 2
   elif state == State.COOR2:  # Pick position
      # Pick the cup
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H]           |
|                         |
|                         |
|                         |
|                         |
|          [Cup]* Down    |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.PICK
      tag = 3
   elif state == State.COOR3:  # Move back up, to avoid opstacles
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H]           |
|                         |
|                         |
|                         |
|                         |
|          [Cup]* Up      |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.COOR4
      tag = 4
   elif state == State.COOR4:  # Intermediate position
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H][Cup]* Up  |
|                         |
|                         |
|                         |
|                         |
|                         |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.COOR5
      tag = 5
   elif state == State.COOR5:  # Place position
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H][Cup]* Down|
|                         |
|                         |
|                         |
|                         |
|                         |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.PLACE
      tag = 6
   elif state == State.COOR6:  # Final position after placing
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H][Cup]* Up  |
|                         |
|                         |
|                         |
|                         |
|                         |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.END
      tag = 7
   elif state == State.END:  # End position
      print("""
+-------------------------+
|                         |
|                         |
|        Home Point       |
|           [H][Cup]* Up  |
|                         |
|                         |
|                         |
|                         |
|                         |
|                         |
+-------------------------+

Legend:
[H]   - Home Point
[Cup] - Cup
*     - End Effector Location
""")
      state = State.RETURN1
      tag = 8
   #FOR RETURNING OBJECT REVERSE LOGIC
   elif state == State.RETURN1:  # Return path
      state = State.RETURN2
      tag = 9
   elif state == State.RETURN2:  # Further return path
      state = State.PICK2
      tag = 10
   elif state == State.RETURN3:  # Continue return path
      state = State.RETURN4
      tag = 11
   elif state == State.RETURN4:  # Further return path
      state = State.RETURN5
      tag = 12
   elif state == State.RETURN5:  # Final return path
      state = State.PLACE2
   elif state == State.RETURN6:  # End return path
      state = State.RETURN7
      tag = 13

   # Start the movement transaction and move the robot to the specified coordinates
   transaction = conn.start_transaction() # Start a new transaction
   transaction.move_to_point_ptp(arr, 0.80, 200)  # Move to point with specific velocity and acceleration
   transaction.set_queue_tag(tag, wait_for_completion=True)  # Set a queue tag for each movement
   transaction.wait_for_queue_tag(tag)  # Wait for movement completion
   await transaction.submit()  # Submit the transaction

# -----------------------------------------------------------------------------
# Function: test_connection_and_movement
# -----------------------------------------------------------------------------
# Purpose:
# This function tests the connection to the robot and manages its movement 
# based on the current state. It continuously checks the connection status
# and directs the robot to perform its tasks (like picking up or placing an object).
# It also controls the gripper based on the type of object being manipulated.
#
# Parameters:
# - robot_ip: The IP address of the robot, used to establish the connection.
#
# Scenarios:
# This function is used as the main loop for robot operation, ensuring that the
# robot remains connected and operational. It cycles through the various states
# and executes the corresponding movements.
#
#GRIPPER USES THE SVR CONNECTION
#ROBOT ARM USES THE SCT CONNECTION

async def test_connection_and_movement(robot_ip):
   global state
   global target_object
   
   while True:
      start = time.time()
      status = {'SCT': 'offline', 'SVR': 'offline', 'STA': 'offline'}  # Initialize connection statuses

      # Check SVR connection (always active)
      try:
         async with techmanpy.connect_svr(robot_ip=robot_ip, conn_timeout=1) as conn:
               status['SVR'] = 'online'
               await conn.get_value('Robot_Model')  # Verify connection by fetching robot model
               status['SVR'] = 'connected'
               
               # Adjust gripper based on the current state
               if state ==  State.HOME:
                  await conn.set_value('g_release', '0')  # Reset gripper
                  print('Setting g release')
               elif state == State.PICK:
                  # Set gripper values based on the target object
                  if target_object == "orange":
                     await conn.set_value('g_release', '20')  # Orange gripper value
                  elif target_object == "vase":
                     await conn.set_value('g_release', '38')  # Vase gripper value
                  elif target_object == "cup":
                     await conn.set_value('g_release', '33')  # Cup gripper value
                  elif target_object == "teddy bear":
                     await conn.set_value('g_release', '50')  # Teddy bear gripper value
                  print('Setting g release')
                  state = State.COOR3
               elif state == state.PICK2:
                  # Set gripper values for return state based on the target object
                  if target_object == "orange":
                     await conn.set_value('g_release', '20')  # Orange gripper value
                  elif target_object == "vase":
                     await conn.set_value('g_release', '38')  # Vase gripper value
                  elif target_object == "cup":
                     await conn.set_value('g_release', '33')  # Cup gripper value
                  elif target_object == "teddy bear":
                     await conn.set_value('g_release', '50')  # Teddy bear gripper value
                  print('Setting g release')
                  state = State.RETURN3
               elif state == State.PLACE:
                  await conn.set_value('g_release', '0')  # Release gripper after placing object
                  print('Setting g release to 0')
                  state = State.COOR6
               elif state == State.PLACE2:
                  await conn.set_value('g_release', '0')  # Final release of gripper
                  print('Setting g release')
                  state = State.RETURN6

               time.sleep(4)  # Small delay for safety
         
      except TechmanException: 
         pass  # Handle any exceptions during connection

      # Check SCT connection (only active when inside listen node)
      try:
         async with techmanpy.connect_sct(robot_ip=robot_ip, conn_timeout=1) as conn:
               status['SCT'] = 'online'
               await conn.resume_project()  # Resume project on robot
               status['SCT'] = 'connected'

               # Move robot based on the current state
               if state == State.HOME:
                  print("Moving to home")
                  await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])  # Move to home position
               elif state == State.COOR1:
                  print("Moving to coord1")  # Position before picking
                  await move(conn,itemupcoor)
               elif state == State.COOR2:
                  print("Moving to coord2")  # Picking position
                  await move(conn,itemcoor)
               elif state == State.COOR3:
                  print("Moving to coord3")
                  await move(conn,itemupcoor)
               elif state == State.COOR4:
                  print("Moving to coord4")
                  await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])  # Move to home position
               elif state == State.COOR5:
                  print("Moving to coord5")
                  if target_object == "orange":
                     await move(conn,[184.16, -390.07, itemcoor[2]-60, 90.65, -0.49, 0])  # Adjusted Z for orange
                  else:
                     await move(conn,[184.16, -390.07, itemcoor[2], 90.65, -0.49, 0])  # Normal move
               elif state == State.COOR6:
                  print("Moving to coord6")
                  await move(conn,[184.16, -350.07, 315.64, 90.65, -0.49, 0])  # Final position
               elif state == State.END:
                  print("Moving to HOME")
                  await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])  # Return to home position
               
      except TechmanException: 
         pass  # Handle any exceptions during connection

      # Check STA connection (only active when running project)
      try:
         async with techmanpy.connect_sta(robot_ip=robot_ip, conn_timeout=1) as conn:
               status['STA'] = 'online'
               await conn.is_listen_node_active()  # Check if listen node is active
               status['STA'] = 'connected'
      except TechmanException: 
         pass  # Handle any exceptions during connection

      # Print the status of the connections with color coding
      def colored(status):
         if status == 'online': return f'\033[96m{status}\033[00m'
         if status == 'connected': return f'\033[92m{status}\033[00m'
         if status == 'offline': return f'\033[91m{status}\033[00m'
      print(f'SVR: {colored(status["SVR"])}, SCT: {colored(status["SCT"])}, STA: {colored(status["STA"])}')

# -----------------------------------------------------------------------------
# Main Script Execution
# -----------------------------------------------------------------------------
# Purpose:
# This section is the entry point of the script. It expects a command-line argument
# (the robot's IP address) and initiates the connection and movement test by calling
# the `test_connection_and_movement` function. It handles interruptions gracefully.
#
# Scenarios:
# This block is executed when the script is run directly from the command line.
# It ensures that the correct arguments are provided and starts the main operation.

if __name__ == '__main__':
   if len(sys.argv) == 2:
      try:
         # Run the main connection and movement function with the provided robot IP address
         asyncio.run(test_connection_and_movement(sys.argv[1]))
      except KeyboardInterrupt: 
         print()  # Handle KeyboardInterrupt to terminate gracefully
   else:
      print(f'usage: {sys.argv[0]} <robot IP address>')  # Print usage instructions if arguments are missing
