#!/usr/bin/env python3
import gpio
import sys
import config as cf
import time
import threading
import global_storage as gs

class CarGuard(threading.Thread):

    def __init__(self, stop_on_distance_sensor_covered=True, stop_on_button_4=True, stop_duration=None):
        threading.Thread.__init__(self)
        self.stop_on_distance_sensor_covered = stop_on_distance_sensor_covered
        self.stop_on_button_4 = stop_on_button_4
        self.stop_duration = stop_duration
        self.stop_time_begin = time.time()

    def run(self):
        while not gs.exit_signal:
            condition1 = (not gs.button_ss2) and self.stop_on_distance_sensor_covered
            condition2 = gs.button_4 and self.stop_on_button_4
            if condition1 or condition2:
                gs.emergency_stop = True
                self.stop_time_begin = time.time()
            if self.stop_duration is not None:
                if time.time() - self.stop_time_begin > self.stop_duration:
                    gs.emergency_stop = False