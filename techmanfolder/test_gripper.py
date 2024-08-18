#!/usr/bin/env python

import sys
import asyncio
import time

import techmanpy
from techmanpy import TechmanException

from pyModbusTCP.client import ModbusClient

executionTime=0
MODBUS_SERVER_IP="192.168.250.31"

async def test_gripper(conn):
    try:
        c = ModbusClient(host=MODBUS_SERVER_IP, port=5891, auto_open=True, unit_id=9)
        print("Host found")
    except ValueError:
        print("Error with host.")

    # managing TCP sessions with call to c.open()/c.close()
    c.open()

    #Wait for the connection to establish
    time.sleep(5)

    #Write output register to request and activation
    response=c.write_multiple_registers(0,[0b0000000100000000,0,0])
    print(response)

    #Give some time for the gripper to activate
    print("Gripper activate")
    time.sleep(5)

    response=c.write_multiple_registers(0,[0b0000100100000000,0b0000000011111111,0b1111111111111111])

    #Give some time for the gripper to reach the desired position
    print("Close Gripper")
    time.sleep(3)

    #close connection
    c.close()
    exit()

async def move(conn):
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
      except TechmanException: pass

      # Check SCT connection (only active when inside listen node)
      try:
         async with techmanpy.connect_sct(robot_ip=robot_ip, conn_timeout=1) as conn:
            status['SCT'] = 'online'
            await conn.resume_project()
            status['SCT'] = 'connected'
            await test_gripper(conn)  # Test gripper
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