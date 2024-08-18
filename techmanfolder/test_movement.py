#!/usr/bin/env python

import sys
import asyncio
import time

import techmanpy
from techmanpy import TechmanException

async def move(conn):
   # First point
   await conn.move_to_joint_angles_ptp([10, -10, 10, -10, 10, -10], 0.10, 200)

   
async def test_connection_and_movement(robot_ip):
   while True:
      start = time.time()
      status = {'SCT': 'offline', 'SVR': 'offline', 'STA': 'offline'}

      # Check SVR connection (should be always active)
      try:
         async with techmanpy.connect_svr(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['SVR'] = 'online'
            await conn.get_value('Robot_Model')
            status['SVR'] = 'connected'

            print("Getting gripper...")
            param = await conn.get_value('g_release')
            print(param)
            await conn.set_value('g_release', '100')
            # if param == 100:
            #    await conn.set_value('g_release', '0')
            # else:
            #    await conn.set_value('g_release', '100')

            # if param == 0 and param2 == 100:
            #    await conn.set_value('g_release', '100')
            #    await conn.set_value('g_grip', '0')
            # else: 
            #     await conn.set_value('g_release', '0')
            #     await conn.set_value('g_grip', '100')
            # # await conn.set_value('g_release', '0')
            # param = await conn.get_value('g_release')
            # print(param)

            # param2 = await conn.get_value('g_grip')
            # print(param2)

            
      except TechmanException: pass

      # Check SCT connection (only active when inside listen node)
      try:
         async with techmanpy.connect_sct(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['SCT'] = 'online'
            await conn.resume_project()
            status['SCT'] = 'connected'
            
            await move(conn)  # Movement
            
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
      elapsed = time.time() - start
      if elapsed < 2: time.sleep(2 - elapsed)

if __name__ == '__main__':
   if len(sys.argv) == 2:
      try: asyncio.run(test_connection_and_movement(sys.argv[1]))
      except KeyboardInterrupt: print() # terminate gracefully
   else: print(f'usage: {sys.argv[0]} <robot IP address>')
