import sys, getopt
sys.path.append('.')
import RTIMU
import os.path
import time
import math
import config as cf
#done
print("Using settings file " + cf.MPU9250_SETTING_FILE)
if not os.path.exists(cf.MPU9250_SETTING_FILE):
  print("Settings file does not exist, will be created")

s = RTIMU.Settings(cf.MPU9250_SETTING_FILE)
imu = RTIMU.RTIMU(s)

print("IMU Name: " + imu.IMUName())

if (not imu.IMUInit()):
    print("IMU Init Failed")
    sys.exit(1)
else:
    print("IMU Init Succeeded")

# this is a good time to set any fusion parameters

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

poll_interval = imu.IMUGetPollInterval()
print("Recommended Poll Interval: %dmS\n" % poll_interval)

while True:
  if imu.IMURead():
    # x, y, z = imu.getFusionData()
    # print("%f %f %f" % (x,y,z))
    data = imu.getIMUData()
    fusionPose = data["fusionPose"]
    print("r: %f p: %f y: %f" % (math.degrees(fusionPose[0]), 
        math.degrees(fusionPose[1]), math.degrees(fusionPose[2])))
    time.sleep(poll_interval*1.0/1000.0)