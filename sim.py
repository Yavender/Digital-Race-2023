import json
import cv2
import numpy as np

import asyncio
import websockets
from PIL import Image
import base64
from io import BytesIO

from Road_Line_Segmentation.video_demo import segment

from onnx import detect_traffic_signs, model

IMAGE_H = 480
IMAGE_W = 640

center = IMAGE_W//2
lane_width = IMAGE_W
left_point = 100
right_point = 220
top_right_point = 0
top_left_point = 0

turnleft = False
turnright = False
gostraight = False
stop = 0
turnleft_count = 0
turnright_count = 0
gostraight_count = 0
duration = 0


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(blur, 30, 80)
    return canny


def birdview(image):
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, IMAGE_H // 2], [IMAGE_W, IMAGE_H // 2]])
    dst = np.float32([[100, IMAGE_H], [IMAGE_W - 100, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    return warped_image


def white_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 255])
    upper_white = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_filter = cv2.bitwise_and(image, image, mask=mask)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_HSV2BGR)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_BGR2GRAY)
    return mask


def houglines(image, draw):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 30, minLineLength=3, maxLineGap=150)
    hough = np.zeros([IMAGE_H, IMAGE_W, 3])
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return hough


def left_right_point(image, draw, line=0.85):
    global center, lane_width, left_point, right_point

    red_line_y = int(IMAGE_H * line)
    red_line = image[red_line_y, :]

    find_left = False
    find_right = False

    for x in range(center - 10, 0, -1):
        if red_line[x].all() > 0:
            left_point = x
            find_left = True
            break
    for x in range(center + 10, IMAGE_W):
        if red_line[x].all() > 0:
            right_point = x
            find_right = True
            break

    if not find_left and find_right:
        left_point = right_point - lane_width
    if not find_right and find_left:
        right_point = left_point + lane_width

    center = (right_point + left_point) // 2
    lane_width = right_point - left_point

    cv2.line(draw, (left_point, red_line_y), (right_point, red_line_y), (0, 0, 255), 1)
    cv2.circle(draw, (left_point, red_line_y), 5, (255, 0, 0), -1)
    cv2.circle(draw, (right_point, red_line_y), 5, (0, 255, 0), -1)
    cv2.circle(draw, (center, red_line_y), 5, (0, 0, 255), -1)


def calculate_speed_angle(signs):
    global turnleft_count, turnright_count, turnleft, turnright, gostraight_count, gostraight, duration, speed, steer, stop
    speed = 0.3
    center_diff = IMAGE_W // 2 - center
    steer = -float(center_diff * 0.04)

    speed -= abs(steer) * 1.3
    if speed <= 0.12:
        speed = 0.12
        
    if not turnleft and not turnright and not gostraight:
        if "left" in signs:
            turnleft = True
        if "right" in signs:
            turnright = True
        if "straight" in signs:
            gostraight = True
    if "stop" in signs:
        stop = 1
        duration = 100
    if turnleft_count > 0:
        turnleft_count += 1
        speed = 0.12
        steer = -1
        if turnleft_count >= duration:
            turnleft_count = 0
            duration = 0
    if turnright_count > 0:
        turnright_count += 1
        speed = 0.12
        steer = 1
        if turnright_count >= duration:
            turnright_count = 0
            duration = 0
    if gostraight_count > 0:
        gostraight_count += 1
        steer = 0.12
        if gostraight_count >= duration:
            gostraight_count = 0
            duration = 0  
    if stop > 0:
        stop += 1
        speed = 0
        if stop >= duration:
            stop = 0
            duration = 0

    return speed, steer

async def echo(websocket, path):
    async for message in websocket:
        # Get image from car
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Birdview transformation and Canny
        image_birdview = birdview(image)
        canny_image = canny(image)

        # Prepare images to be draw on
        point_image = image.copy()
        traffic_image = image.copy()
        hough_image = image.copy()

        # Prepare images for algorithm
        w_filter = white_filter(image)
        #hough = houglines(w_filter, hough_image)
        
        segment_image = segment(image)

        # Algorithm
        left_right_point(segment_image, point_image)
        #signs = detect_traffic_signs_without_model(image, traffic_image)
        signs = detect_traffic_signs(image, model, point_image)
        
        segment_image = segment(image)

        # Mesure and decide speed and angle
        speed, steer = calculate_speed_angle(signs)
        
        # Show images
        #cv2.imshow("image", image)
        cv2.imshow("canny", canny_image)
        #cv2.imshow("white filter", w_filter)
        #cv2.imshow("hough", hough_image)
        cv2.imshow("point", point_image)
        #cv2.imshow("traffic", traffic_image)
        cv2.imshow("segment_image", segment_image)

        cv2.waitKey(1)
        message = json.dumps({"throttle": speed, "steering": steer})
        print(message)
        await websocket.send(message)

async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()

asyncio.run(main())
