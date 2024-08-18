#!/usr/bin/env python

from pyModbusTCP.client import ModbusClient
import time

#Initialize variables
##################
executionTime=0
MODBUS_SERVER_IP="192.168.250.31"

#Process initialization
##################
#communication
# TCP auto connect on first modbus request
try:
    c = ModbusClient(host=MODBUS_SERVER_IP, port=5891, auto_open=True)
    bits = c.read_coils
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