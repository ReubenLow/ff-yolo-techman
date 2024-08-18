import sys
import asyncio
import time
from pymodbus.client.sync import ModbusTcpClient
import techmanpy
from techmanpy import TechmanException

# Replace with your gripper's IP address
gripper_ip = '192.168.250.31'
client = ModbusTcpClient(gripper_ip)

async def test_gripper(conn):
   # Connect to the gripper
   connection = client.connect()
   if connection:
      print("Connected to the gripper")

      # Activate the gripper
      client.write_register(1000, 0x01)  # Write 1 to rACT bit to activate the gripper

      # Command the gripper to go to the requested position with speed and force settings
      client.write_registers(1001, [0xFF, 0xFF, 0xFF])  # Fully close with max speed and force

      # Verify the gripper status
      response = client.read_holding_registers(2000, 3)  # Read status registers
      print(f"Gripper status: {response.registers}")

      # Close the connection
      client.close()
   else:
      print("Failed to connect to the gripper")

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
