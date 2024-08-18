#!/usr/bin/env python

import sys
import asyncio
import time
from enum import Enum
import techmanpy
from techmanpy import TechmanException
import copy
import json

# Specify the path to the JSON file, change accordingly to specify your directory
json_file_path = "/home/reuben/techmanpy/coordinates.json"
obj_file_path = "/home/reuben/techmanpy/object.json"

# Load the coordinates from JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Load specified object from JSON file
with open(obj_file_path, "r") as file:
    target_object = json.load(file)

# Extract the values and create an array (list)
itemcoor = list(data.values())
itemupcoor = copy.deepcopy(itemcoor)
itemupcoor[2] += 200 #offset in z, prev value 100
itemupcoor[0] -= 38 #offset in x 
itemcoor[0] -= 38 #offset in x 

# Not for orange
# itemupcoor[1] -= 20 #offset in y 
# itemcoor[1] -= 20 #offset in y 

# Orange x, y values
if target_object == "orange":
   itemupcoor[1] -= 35 #offset in y 
   itemcoor[1] -= 35 #offset in y 
elif target_object == "vase":
   itemupcoor[1] -= 40 #offset in y 
   itemcoor[1] -= 40 #offset in y
elif target_object == "cup":
   itemupcoor[1] -= 35 #offset in y 
   itemcoor[1] -= 35 #offset in y
else:
   itemupcoor[1] -= 20 #offset in y 
   itemcoor[1] -= 20 #offset in y 



# Output the array
print(itemcoor)
print(itemupcoor)


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


# state = State.START
# state = State.COOR1
state = State.HOME


async def move(conn, arr):
   global state
   tag = 1
   if state == State.HOME: #Home pos
      state = State.COOR1
      tag = 1
   elif state == State.COOR1: # coor1 pos before picking
      state = State.COOR2
      tag = 2
   elif state == State.COOR2: #coor2 pick
      state = State.PICK
      tag = 3
   elif state == State.COOR3: #COOR3
      state = State.COOR4
      tag = 4
   elif state == State.COOR4: #COOR4 
      state = State.COOR5
      tag = 5
   elif state == State.COOR5: #COOR5 PLACE
      state = State.PLACE
      # state = State.NEUTRAL
      tag = 6
   elif state == State.COOR6:
      # state = State.HOME
      state = State.END
      tag = 7
   elif state == State.END:
      # state = State.HOME
      state = State.RETURN1
      tag = 8
   elif state == State.RETURN1:
      state = State.RETURN2
      tag = 9
   elif state == State.RETURN2:
      state = State.PICK2
      tag = 10
   elif state == State.RETURN3:
      state = State.RETURN4
      tag = 11
   elif state == State.RETURN4:
      state = State.RETURN5
      tag = 12
   elif state == State.RETURN5:
      state = State.PLACE2
   elif state == State.RETURN6:
      state = State.RETURN7
      tag = 13
   # transaction = conn.start_transaction()
   # transaction.set_queue_tag(tag, wait_for_completion=True)
   # transaction.move_to_joint_angles_ptp(arr, 0.80, 200)
   # transaction.wait_for_queue_tag(tag)
   # await transaction.submit()'

   
   transaction = conn.start_transaction()
   #transaction.move_to_joint_angles_ptp(arr, 0.80, 200) #for joing angle movement
   # transaction.set_tcp('TCP_Name')
   transaction.move_to_point_ptp(arr, 0.80, 200)
   transaction.set_queue_tag(tag, wait_for_completion=True)
   transaction.wait_for_queue_tag(tag)
   await transaction.submit()

   



# async def move(conn):
#    # First point
# #    await conn.move_to_joint_angles_ptp([10, -10, 10, -10, 10, -10], 0.40, 200)

#    await conn.move_to_joint_angles_ptp([91.49, -43.85, -3.03, -47.64, -96.58, -10.45], 0.50, 200)

   
async def test_connection_and_movement(robot_ip):
   global state
   global target_object
   while True:
      start = time.time()
      status = {'SCT': 'offline', 'SVR': 'offline', 'STA': 'offline'}

      # Check SVR connection (should be always active)
      try:
         async with techmanpy.connect_svr(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['SVR'] = 'online'
            await conn.get_value('Robot_Model')
            status['SVR'] = 'connected'

            # print("Getting gripper...")
            # param = await conn.get_value('g_release')
            # print(param)
            if state ==  State.HOME:
               await conn.set_value('g_release', '0')
               print('Setting g release')
            elif state == State.PICK:
               if target_object == "orange":
                  await conn.set_value('g_release', '20') # orange value
               elif target_object == "vase":
                  await conn.set_value('g_release', '38') # vase value, prev value 40
               elif target_object == "cup":
                  await conn.set_value('g_release', '33') # glass cup value
               elif target_object == "teddy bear":
                  await conn.set_value('g_release', '50') # teddy value
               print('Setting g release')
               # time.sleep(1)
               state = State.COOR3
            elif state == state.PICK2:
               if target_object == "orange":
                  await conn.set_value('g_release', '20') # orange value
               elif target_object == "vase":
                  await conn.set_value('g_release', '38') # vase value, prev value 40
               elif target_object == "cup":
                  await conn.set_value('g_release', '33') # glass cup value
               elif target_object == "teddy bear":
                  await conn.set_value('g_release', '50') # teddy value
               print('Setting g release')
               state = State.RETURN3
            elif state == State.PLACE:
               await conn.set_value('g_release', '0')
               print('Setting g release to 0')
               # time.sleep(1)
               state = State.COOR6
            elif state == State.PLACE2:
               await conn.set_value('g_release', '0')
               print('Setting g release')
               state = State.RETURN6


            time.sleep(4)

            
      except TechmanException: pass

      # Check SCT connection (only active when inside listen node)
      try:
         async with techmanpy.connect_sct(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['SCT'] = 'online'
            await conn.resume_project()
            status['SCT'] = 'connected'
            
            # await move(conn)  # Movement
            # await conn.set_queue_tag(1, True)
            # await conn.wait_for_queue_tag(1)
            # state = 1

            # Create queue transaction
            if state == State.HOME:
               print("Moving to home")
               # await move(conn,[184.16, -390.07, 215.64, 90.65, -0.49, -4.85]) #done
               await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])
            elif state == State.COOR1:
               print("Moving to coord1")     # Pos before picking
               await move(conn,itemupcoor)
            elif state == State.COOR2:
               print("Moving to coord2")
               await move(conn,itemcoor)#done
               # break;
            elif state == State.COOR3:
               print("Moving to coord3")
               await move(conn,itemupcoor)
            elif state == State.COOR4:
               print("Moving to coord4")
               # await move(conn,[184.16, -390.07, 215.64, 90.65, -0.49, -4.85]) #done
               await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0]) # home
            elif state == State.COOR5:
               print("Moving to coord5")
               if target_object == "orange":
                  await move(conn,[184.16, -390.07, itemcoor[2]-60, 90.65, -0.49, 0]) #done
               else:
                  await move(conn,[184.16, -390.07, itemcoor[2], 90.65, -0.49, 0]) #done
               # await move(conn,[110.07, -28.04, -119.94, -31.32, 114.72, -179.11])#done, object down
               
            elif state == State.COOR6:
               print("Moving to coord6")
               # await move(conn,[184.16, -390.07, 215.64, 90.65, -0.49, -4.85]) #done
               await move(conn,[184.16, -350.07, 315.64, 90.65, -0.49, 0])
            elif state == State.END:
               print("Moving to HOME")
               # await move(conn,[184.16, -390.07, 215.64, 90.65, -0.49, -4.85]) #done
               await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])
      

            # elif state == State.RETURN1:  # Return object to original position
            #    await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])
            # elif state == state.RETURN2:  # Pick up
            #    print("Picking object")
            #    if target_object == "orange":
            #       await move(conn,[184.16, -390.07, itemcoor[2]-60, 90.65, -0.49, 0]) #done
            #    else:
            #       await move(conn,[184.16, -390.07, itemcoor[2], 90.65, -0.49, 0]) #done

            # elif state == State.RETURN3:  # Go up
            #    await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0]) # home
            # elif state == State.RETURN4:  # Go to location
            #    await move(conn,itemupcoor)      # before release
            # elif state == State.RETURN5:  # Go down, release
            #    await move(conn,itemcoor)#done
            # elif state == State.RETURN6:  # Go back up
            #    await move(conn,itemupcoor)
            # elif state == State.RETURN7:  # Go home
            #    print("Going home")
            #    await move(conn,[184.16, -390.07, 315.64, 90.65, -0.49, 0])

            
      except TechmanException: pass

      # Check STA connection (only active when running project)
      try:
         async with techmanpy.connect_sta(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['STA'] = 'online'
            await conn.is_listen_node_active()
            status['STA'] = 'connected'
      except TechmanException: pass

      # Print status
      def colored(status):
         if status == 'online': return f'\033[96m{status}\033[00m'
         if status == 'connected': return f'\033[92m{status}\033[00m'
         if status == 'offline': return f'\033[91m{status}\033[00m'
      print(f'SVR: {colored(status["SVR"])}, SCT: {colored(status["SCT"])}, STA: {colored(status["STA"])}')

      # Sleep 2 seconds (at most)
      # elapsed = time.time() - start
      # if elapsed < 2: time.sleep(2 - elapsed)

if __name__ == '__main__':
   if len(sys.argv) == 2:
      try: asyncio.run(test_connection_and_movement(sys.argv[1]))
      except KeyboardInterrupt: print() # terminate gracefully
   else: print(f'usage: {sys.argv[0]} <robot IP address>')
   