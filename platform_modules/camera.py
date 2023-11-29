#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from primesense import openni2#, nite2
from primesense import _openni2 as c_api
import config as cf
import time
import threading
import global_storage as gs
import os
import pathlib
import logging
from utils.queue_handle import *

class Camera(threading.Thread):
    """This thread initialize 2 other threads (rgb and depth grabbers)
    """

    def __init__(self):
        threading.Thread.__init__(self)

        # Init device
        openni2.initialize(cf.OPENNI_PATH) #
        self.device = openni2.Device.open_any()

        self.record_videos = gs.record_videos
        self.frame_count = 0

    def run(self):

        # Init RGB thread
        rgb_grabber = RGBCamera(self.device)
        rgb_grabber.start()

        # Init depth thread
        depth_grabber = DepthCamera(self.device)
        depth_grabber.start()

        # Create logger with 'camera'
        self.logger = logging.getLogger('camera')
        self.logger.setLevel(logging.INFO)

        while not gs.exit_signal:
            time.sleep(1/20.0)
            if self.record_videos != gs.record_videos:
                if gs.record_videos: # Turn on video recording

                    print("Start recording...")

                    folder_name = time.strftime("%Y%m%d-%H%M%S")

                    video_folder = os.path.join(cf.RECORDING_PATH, folder_name)
                    pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)

                    gs.rgb_video_file = os.path.join(video_folder, "rgb_" + folder_name + ".avi")
                    gs.depth_video_file = os.path.join(video_folder, "depth_" + folder_name + ".avi")

                    self.logger.handlers = []
                    fh = logging.FileHandler(os.path.join(video_folder, "log.txt"))
                    formatter = logging.Formatter('%(asctime)s %(message)s')
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)

                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    self.rgb_out_file = cv2.VideoWriter(gs.rgb_video_file, fourcc, 20.0, (320,240))
                    self.depth_out_file = cv2.VideoWriter(gs.depth_video_file, fourcc, 20.0, (320, 240))

                    self.frame_count = 0

                else: # Stop video recording

                    print("Stop recording...")

                    self.rgb_out_file.release()
                    self.depth_out_file.release()

                self.record_videos = gs.record_videos

            elif self.record_videos: # Write video to file

                # Write rgb video
                self.rgb_out_file.write(gs.rgb_frames.get())

                # Write depth video
                img_rgb = cv2.cvtColor(gs.depth_frames.get(),cv2.COLOR_GRAY2BGR)
                cvuint8 = cv2.convertScaleAbs(img_rgb, alpha=(255.0/65535.0))
                self.depth_out_file.write(cvuint8)

                self.logger.info("FrameID=" + str(self.frame_count) + " Speed=" + str(gs.speed) + " Steer=" + str(gs.steer))
                self.frame_count += 1


        # Wait threads to exit
        rgb_grabber.join()
        depth_grabber.join()

        print("Camera thread")


class RGBCamera(threading.Thread):

    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device
    def run(self):

        # Start camera stream
        rgb_stream = self.device.create_color_stream()
        rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
        rgb_stream.start()

        # Get image continuously
        while not gs.exit_signal:
            img = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
            bgr   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            put_to_queue_no_wait_no_block(bgr, gs.rgb_frames)

class DepthCamera(threading.Thread):

    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device
    def run(self):

        print("Start depth image thread")

        # Start camera stream
        depth_stream = self.device.create_depth_stream()
        depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 320, resolutionY = 240, fps = 30))
        depth_stream.start()

        # Get image continuously
        while not gs.exit_signal:
            frame = depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            img = np.frombuffer(frame_data, dtype=np.uint16)
            img.shape = (240, 320)
            img = cv2.flip(img,1)

            put_to_queue_no_wait_no_block(img, gs.depth_frames)

        print("Exiting from Depth image grabber")