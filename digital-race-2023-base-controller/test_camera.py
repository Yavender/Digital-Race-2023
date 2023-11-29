import cv2
from platform_modules.camera import Camera
import global_storage as gs
import signal
import sys
# Done
# Init camera
camera = Camera()
camera.start()

# Grab images and show
while not gs.exit_signal:
    if not gs.rgb_frames.empty():
        rgb = gs.rgb_frames.get()
        cv2.imshow("RGB", rgb)
        cv2.waitKey(1)
    if not gs.depth_frames.empty():
        depth = gs.depth_frames.get()
        cv2.imshow("Depth", depth)
        cv2.waitKey(1)

camera.join()
