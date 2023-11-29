import signal
import sys
from platform_modules.motor_controller import MotorController
from utils.keyboard_getch import _Getch
import global_storage as gs
import config as cf
import time

# Init motor controller
mc = MotorController()
mc.start()

# Find throttle neutral
mode = 0
mode = int(input("put 1 to forward, put 2 to reverse"))
if mode == 1: #with forward
    for i in range (0, 200):
        cf.THROTTLE_NEUTRAL = cf.THROTTLE_NEUTRAL - 5
        gs.speed = 0
        for i in range(0, 18):
            
            gs.speed = min(cf.MAX_SPEED, gs.speed + 1)
            print(gs.speed)
            print("-------------")
            time.sleep(0.2)
        gs.speed = 0
        print(cf.THROTTLE_NEUTRAL)
        print("----------------------------------")
        time.sleep(1)
if mode == 2:
    for i in range (0, 200):
        cf.THROTTLE_NEUTRAL = cf.THROTTLE_NEUTRAL + 5
        gs.speed = 0
        for i in range(0, 200):
            cf.THROTTLE_MAX_REVERSE = cf.THROTTLE_MAX_REVERSE - 5
            for i in range(0, 16):
                
                gs.speed = max(-cf.MAX_SPEED, gs.speed - 1)
                print(gs.speed)
                print("****")
                time.sleep(0.5)
            gs.speed = 0
            print(cf.THROTTLE_MAX_REVERSE)
            print("++++++++++++++++++")
            time.sleep(1)
        print(cf.THROTTLE_NEUTRAL)
        print("----------------------------------")
        time.sleep(1)

# Manual control using keyboard
mc.join()
