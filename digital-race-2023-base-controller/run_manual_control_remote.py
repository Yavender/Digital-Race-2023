import signal
import sys
import time
import _thread
from platform_modules.motor_controller import MotorController
from platform_modules.lcd_driver import LCD
from platform_modules.button_reader import ButtonReader
from platform_modules.car_guard import CarGuard
from platform_modules.camera import Camera
from platform_modules.remote_control.remote_controller_udp import RemoteControllerUDP
from utils.keyboard_getch import _Getch
import global_storage as gs
import config as cf


# Init camera
camera = Camera()
camera.start()

# Init LCD
lcd = LCD(cf.LCD_ADDRESS)

# Init button reader
button_reader = ButtonReader()
button_reader.start()

# Init motor controller
motor_controller = MotorController()
motor_controller.start()

# Init remote controller
remote_controller = RemoteControllerUDP()
remote_controller.start()


# Car guard
# Stop car when hitting obstacle or when user presses button 4
guard = CarGuard(stop_on_distance_sensor_covered=False)
guard.start()

# UI thread
def ui_thread():
    # Save last value of buttons
    last_button_1 = False
    last_button_2 = False
    last_button_3 = False
    last_button_4 = False
    last_button_ss1 = False
    last_button_ss2 = False
    last_time_update_screen = time.time()

    while True:
        # Determine pressed buttons
        pressed_buttons = []
        if gs.button_1 and not last_button_1:
            pressed_buttons.append("1")
        if gs.button_2 and not last_button_2:
            pressed_buttons.append("2")
        if gs.button_3 and not last_button_3:
            pressed_buttons.append("3")
        if gs.button_4 and not last_button_4:
            pressed_buttons.append("4")
        if gs.button_ss1 and not last_button_ss1:
            pressed_buttons.append("S1")
        if gs.button_ss2 and not last_button_ss2:
            pressed_buttons.append("S2")

        if any([s for s in pressed_buttons if "1" == s]):
            gs.record_videos = not gs.record_videos

        if any([s for s in pressed_buttons if "3" == s]):
            gs.emergency_stop = False

        if time.time() - last_time_update_screen > 1:
            lcd.lcd_clear()
            lcd.lcd_display_string("Manual mode:", 1)
            lcd.lcd_display_string("1:RECORD,4:STOP", 2)
         
            if gs.emergency_stop:
                lcd.lcd_display_string("EMERGENCY!!!", 4)

            if gs.record_videos:
                lcd.lcd_display_string("[.] Recording...", 3)

            last_time_update_screen = time.time()

        

        # Update values
        last_button_1 = gs.button_1
        last_button_2 = gs.button_2
        last_button_3 = gs.button_3
        last_button_4 = gs.button_4
        last_button_ss1 = gs.button_ss1
        last_button_ss2 = gs.button_ss2
        
        time.sleep(0.2)

_thread.start_new_thread(ui_thread, ())

# Manual control using keyboard
getch = _Getch()
print("Use keyboard to control: wasd")
print("Quit: q")
while not gs.exit_signal:
    key = getch()
    if key == "w":
        if gs.speed < 0:
            gs.speed = 0
        else:
            gs.speed = min(cf.MAX_SPEED, gs.speed + 2)
    elif key == "s":
        if gs.speed > 0:
            gs.speed = 0
        else:
            gs.speed = max(-cf.MAX_SPEED, gs.speed - 2)
    elif key == "a":
        if gs.steer > 0:
            gs.steer = 0
        else:
            gs.steer = max(cf.MIN_ANGLE, gs.steer - 5)
    elif key == "d":
        if gs.steer < 0:
            gs.steer = 0
        else:
            gs.steer = min(cf.MAX_ANGLE, gs.steer + 5)
    elif key == "i": # Remove emergency stop state
        gs.emergency_stop = False
    elif key == "v":
        gs.record_videos = not gs.record_videos
        print("Record video: " + str(gs.record_videos))
    elif key == "q":
        gs.exit_signal = True
        break

    print("Speed: {}  Steer: {}".format(gs.speed, gs.steer))

# camera.join()
# motor_controller.join()
# guard.join()
# button_reader.join()
# remote_controller.join()
