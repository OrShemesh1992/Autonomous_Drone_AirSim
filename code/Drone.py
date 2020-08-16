import airsim

import sys
import time
from Drone_Auto import image_proc_algorithm
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync()

# print("Flying a small square box using moveByVelocityZ")

# AirSim uses NED coordinates so negative axis is up.
# z of -15 is 15 meters above the original launch point.
z = -20

# Fly given velocity vector for 5 seconds
duration = 1
speed = 20

vx = speed
vy = 0
client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 0))

while True:
    count = input("get key")
    print(count)
    if float(count) == 0:
        vx = 0
        vy = -speed
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 270))
    elif float(count)==1:
        vx = -speed
        vy = 0
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 180))
    elif float(count) == 2:
        vx = 0
        vy = speed
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 90))
    elif float(count) == 3:
        vx = speed
        vy = 0
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                         airsim.YawMode(False, 0))
    elif float(count) == 4:
        print(client.getMultirotorState().kinematics_estimated.position)
    else:
        break

airsim.wait_key('Press any key to reset to original state')
client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)