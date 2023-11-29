#!/usr/bin/env python3

import Adafruit_PCA9685
import sys
import config as cf
import time
import threading
import global_storage as gs
import math
import signal
usleep = lambda x: time.sleep(x/1000000.0)


class MotorController(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

        # Init controller
        self.pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
        self.pwm.set_pwm_freq(cf.MOTOR_FREQ)
        usleep(10000)

        # Reset state
        self.direction = 0
        pwm_steer_middle = self.value_map(0, cf.MIN_ANGLE, cf.MAX_ANGLE, cf.STEERING_MAX_RIGHT, cf.STEERING_MAX_LEFT)
        self.pwm.set_pwm(cf.STEERING_CHANNEL, 0, pwm_steer_middle)
        self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_NEUTRAL)

    def run(self):
        while not gs.exit_signal:
            self.set_speed(gs.speed)
            self.set_steer(gs.steer)
        self.stop_car_on_exit(None, None)
        print("Exiting from MotorController")

    def set_speed(self, throttle_val):
        if gs.emergency_stop:
            self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_NEUTRAL)
            usleep(187500)
            return

        # Filter too low throttle to protect motors
        if abs(throttle_val) < cf.MIN_ACTIVE_SPEED:
            throttle_val = 0

        if throttle_val > 0:
            if self.direction == -1:
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_MAX_FORWARD)
                usleep(50000)
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_NEUTRAL)
                self.direction = 0
                usleep(50000)
            self.direction = 1
            pwm = self.value_map(throttle_val, 0, 100, cf.THROTTLE_NEUTRAL, cf.THROTTLE_MAX_FORWARD)
            self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, pwm)
            usleep(5000)
        elif throttle_val < 0:
            if self.direction == 1:
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_MAX_REVERSE)
                usleep(50000)
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_NEUTRAL)
                self.direction = 0
                usleep(50000)
            self.direction = -1
            pwm = 4095 - self.value_map( abs(throttle_val), 0, 100 , 4095 - cf.THROTTLE_NEUTRAL , 4095 - cf.THROTTLE_MAX_REVERSE)
            self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, pwm)
            usleep(5000)
        else:
            if self.direction == 1:
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_MAX_REVERSE)
                usleep(50000)
            elif self.direction == -1:
                self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_MAX_FORWARD)
                usleep(50000)
            self.pwm.set_pwm(cf.THROTTLE_CHANNEL, 0, cf.THROTTLE_NEUTRAL)
            usleep(100000)
            self.direction = 0

    def set_steer(self, steer_angle):
        steer_angle =  min(cf.MAX_ANGLE, max(cf.MIN_ANGLE, steer_angle))
        pwm = self.value_map(steer_angle, cf.MIN_ANGLE, cf.MAX_ANGLE, cf.STEERING_MAX_RIGHT, cf.STEERING_MAX_LEFT)
        self.pwm.set_pwm(cf.STEERING_CHANNEL, 0, pwm)
        usleep(2500)
        
    def value_map (self, x, in_min, in_max, out_min, out_max):
        return int( 1.0 * (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min )

    def stop_car_on_exit(self, num, stack):
        gs.emergency_stop = True
        self.set_speed(0)
        self.set_steer(0)
        exit(0)
