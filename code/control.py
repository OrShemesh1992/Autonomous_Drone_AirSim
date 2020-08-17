import airsim
import sys
import time
from Drone_Auto import image_proc_algorithm

#connect and take off drone
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync()

# z of -5 is 5 meters above the original launch point.
z = -5

# Fly given velocity vector for 1 seconds
duration = 1
speed = 2

vx = speed
vy = 0

client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 0))

while True:
    key = input("choose: 4 Turn left,  6 Turn right ,8 straight ,2 backwards ,0 get position,any key to quit: ")
    if float(key) == 4:
        vx = 0
        vy = -speed
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 270))
    elif float(key)==2:
        vx = -speed
        vy = 0
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 180))
    elif float(key) == 6:
        vx = 0
        vy = speed
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 90))
    elif float(key) == 8:
        vx = speed
        vy = 0
        client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                         airsim.YawMode(False, 0))
    elif float(key) == 0:
        print(client.getMultirotorState().kinematics_estimated.position)
    else:
        break

airsim.wait_key('Press any key to reset to original state')
client.armDisarm(False)
client.reset()

# let's quit cleanly
client.enableApiControl(False)