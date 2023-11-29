import signal
import sys
from platform_modules.motor_controller import MotorController
from utils.keyboard_getch import _Getch
import global_storage as gs
import config as cf


# Init motor controller
mc = MotorController()
mc.start()

# Manual control using keyboard
getch = _Getch()
print("Use keyboard to control: wasd")
print("Quit: q")
while not gs.exit_signal:
    key = getch()
    if key == "w":
        if gs.speed < 0:
            gs.speed = max(-cf.MAX_SPEED, gs.speed + 0.5)
        else:
            gs.speed = min(cf.MAX_SPEED, gs.speed + 0.5)
    elif key == "s":
        if gs.speed > 0:
            if gs.speed < 9:
                gs.speed = 0
            gs.speed = min(cf.MAX_SPEED, gs.speed - 0.5)
            
        else:
            gs.speed = max(-cf.MAX_SPEED, gs.speed - 0.5)
    elif key == "a":
        if gs.steer > 0:
            gs.steer = 0
        gs.steer = max(cf.MIN_ANGLE, gs.steer - 5)
    elif key == "d":
        if gs.steer < 0:
            gs.steer = 0
        gs.steer = min(cf.MAX_ANGLE, gs.steer + 5)
    elif key == "q":
        gs.exit_signal = True
        exit(0)
    print("Speed: {}  Steer: {}".format(gs.speed, gs.steer))

mc.join()
