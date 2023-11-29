import signal
import sys
from platform_modules.motor_controller import MotorController
from utils.keyboard_getch import _Getch
import global_storage as gs
import config as cf
from platform_modules.camera import Camera
import json
import cv2
import numpy as np

camera = Camera()
camera.start()
mc = MotorController()
mc.start()

# Manual control using keyboard
getch = _Getch()
print("Use keyboard to control: wasd")
print("Quit: q")

center = 160
lane_width = 96
left_point = 100
right_point = 220

top_right_point = 0
top_left_point = 0

template1 = cv2.imread("traffic/stop.png", 0)
template2 = cv2.imread("traffic/noleft.png", 0)
template3 = cv2.imread("traffic/noright.png", 0)
template4 = cv2.imread("traffic/left.png", 0)
template5 = cv2.imread("traffic/right.png", 0)
template6 = cv2.imread("traffic/straight.png", 0)
templates1 = [template1, template2, template3]
templates2 = [template4, template5, template6]
texts1 = ["stop", "noleft", "noright"]
texts2 = ["left", "right", "straight"]

turnleft = False
turnright = False
gostraight = False
stop = 0
turnleft_count = 0
turnright_count = 0
gostraight_count = 0
duration = 0
speed = 0
steer = 0


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 80)
    return canny


def birdview(image):
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, IMAGE_H // 3], [IMAGE_W, IMAGE_H // 3]])
    dst = np.float32([[100, IMAGE_H], [IMAGE_W - 100, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    return warped_image


def white_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([170, 110, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_filter = cv2.bitwise_and(image, image, mask=mask)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_HSV2BGR)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_BGR2GRAY)
    return white_filter


def houglines(image, draw):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 30, minLineLength=3, maxLineGap=150)
    hough = np.zeros([IMAGE_H, IMAGE_W, 3])
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return hough


def car_filter(image):
    hsv_car = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_car = np.array([0, 0, 50])
    upper_car = np.array([170, 50, 255])
    mask_car = cv2.inRange(hsv_car, lower_car, upper_car)
    car_filter = cv2.bitwise_and(image, image, mask=mask_car)
    car_filter = cv2.cvtColor(car_filter, cv2.COLOR_HSV2BGR)
    car_filter = cv2.cvtColor(car_filter, cv2.COLOR_BGR2GRAY)
    return mask_car


def left_right_point(image, hough, mask_car, draw, line=0.85):
    global center, lane_width, left_point, right_point, has_car_left, has_car_right

    car_line_y = int(IMAGE_H * 0.6)
    car_line_y2 = int(IMAGE_H * 0.85)
    red_line_car = [mask_car[car_line_y, :], mask_car[car_line_y2, :]]

    if gostraight_count != 0:
        image = hough

    red_line_y = int(IMAGE_H * line)
    red_line = image[red_line_y, :]

    find_left = False
    find_right = False
    if turnright_count == 0 and turnleft_count == 0:
        for x in range(center - 10, 0, -1):
            if red_line[x].all() > 0 and center - x > lane_width / 2.2:
                left_point = x
                find_left = True
                break
        for x in range(center + 10, IMAGE_W):
            if red_line[x].all() > 0 and x - center > lane_width / 2.2:
                right_point = x
                find_right = True
                break

    center = (right_point + left_point) // 2

    if turnright_count == 0 and turnleft_count == 0:
        if not find_left or not find_right:
            center = 160

    has_car_left = False
    has_car_right = False
    for line in red_line_car:
        for x in range(center + 1, int(center + lane_width / 4)):
            if line[x].all() == 0:
                has_car_right = True
                print("car right")
                break
        for x in range(center - 1, int(center - lane_width / 4), -1):
            if line[x].all() == 0:
                has_car_left = True
                print("car left")
                break

    cv2.line(draw, (left_point, red_line_y), (right_point, red_line_y), (0, 0, 255), 1)
    cv2.circle(draw, (left_point, red_line_y), 5, (255, 0, 0), -1)
    cv2.circle(draw, (right_point, red_line_y), 5, (0, 255, 0), -1)
    cv2.circle(draw, (center, red_line_y), 5, (0, 0, 255), -1)

    cv2.line(draw, (0, car_line_y), (IMAGE_W, car_line_y), (0, 0, 255), 1)
    cv2.line(draw, (0, car_line_y2), (IMAGE_W, car_line_y2), (0, 0, 255), 1)


def top_right_left_point(hough, draw, line=0.85):
    global top_right_point, top_left_point, turnright_count, turnleft_count, duration, turnleft, turnright, gostraight_count, gostraight

    red_line_y = int(IMAGE_H * line)

    red_line_x2 = min(int(center + lane_width / 2 + lane_width / 3.5), IMAGE_W - 1)
    red_line_x3 = min(int(center - lane_width / 2 - lane_width / 3.5), IMAGE_W - 1)

    red_line_2 = hough[:, red_line_x2]
    red_line_3 = hough[:, red_line_x3]

    distance = 60
    has_right = False
    has_left = False

    for x in range(red_line_y, red_line_y - distance, -1):
        if red_line_2[x].any() > 0:
            top_right_point = x
            has_right = True
            break

    for x in range(red_line_y, red_line_y - distance, -1):
        if red_line_3[x].any() > 0:
            top_left_point = x
            has_left = True
            break

    if has_right and turnright:
        turnright_count = 1
        turnright = False
        duration = 25

    if has_left and turnleft:
        turnleft_count = 1
        turnleft = False
        duration = 25

    if gostraight:
        if has_right or has_left:
            gostraight_count = 1
            gostraight = False
            duration = 20

    cv2.line(draw, (red_line_x2, red_line_y), (red_line_x2, top_right_point), (0, 0, 255), 1)
    cv2.line(draw, (red_line_x3, red_line_y), (red_line_x3, top_left_point), (0, 0, 255), 1)
    cv2.circle(draw, (red_line_x2, top_right_point), 5, (0, 255, 0), -1)
    cv2.circle(draw, (red_line_x3, top_left_point), 5, (255, 0, 0), -1)


def detect_top_point(image, draw, line=0.85):
    global center, lane_width

    IMAGE_H, IMAGE_W = image.shape[:2]
    red_line_x = int(IMAGE_W * center / IMAGE_W) + 10

    red_line_2 = image[:, red_line_x]

    top_point = 0

    for x in range(int(IMAGE_H * line), 0, -1):
        if red_line_2[x].all() == 0:
            top_point = x
            break

    cv2.line(draw, (red_line_x, int(IMAGE_H * line)), (red_line_x, top_point), (0, 0, 255), 2)
    draw = cv2.circle(draw, (red_line_x, top_point), 7, (122, 0, 122), -1)
    return top_point


def filter_signs_by_color(image):
    # RED
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1)
    mask_2 = cv2.inRange(image, lower2, upper2)
    mask_r = cv2.bitwise_or(mask_1, mask_2)
    # BLUE
    lower3, upper3 = np.array([100, 100, 0]), np.array([140, 255, 255])
    mask_b = cv2.inRange(image, lower3, upper3)
    # COMBINE
    mask_final = cv2.bitwise_or(mask_r, mask_b)
    return mask_r, mask_b


def get_boxes_from_mask(mask_r, mask_b):
    bboxes_r = []
    bboxes_b = []
    nccomps = cv2.connectedComponentsWithStats(mask_r, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask_r.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w + h < 30:
            continue
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        if w / h > 1.2 or h / w > 1.2:
            continue
        bboxes_r.append([x, y, w, h])
    nccomps = cv2.connectedComponentsWithStats(mask_b, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask_b.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w + h < 30:
            continue
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        if w / h > 1.2 or h / w > 1.2:
            continue
        bboxes_b.append([x, y, w, h])
    return bboxes_r, bboxes_b


def get_boxes_from_mask_car(mask, draw):
    mask = cv2.bitwise_not(mask)
    bboxes = []
    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if x not in range(20, 300):
            continue
        if x + w > 300:
            continue
        if w + h < 30:
            continue
        if w > 0.5 * im_width or h > 0.5 * im_height:
            continue
        if w / h > 1.3 or h / w > 1.3:
            continue
        bboxes.append([x, y, w, h])
    for bbox in bboxes:
        x, y, w, h = bbox
        if draw is not None:
            cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return bboxes


def detect_traffic_signs_without_model(img, draw=None):
    mask_r, mask_b = filter_signs_by_color(img)
    bboxes_r, bboxes_b = get_boxes_from_mask(mask_r, mask_b)
    list_text = []

    # RED SIGNS
    for bbox in bboxes_r:
        x, y, w, h = bbox

        list_value = []
        crop_img = img[y:y + h, x:x + w]
        center_crop_img = crop_img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        center_crop_img2 = crop_img[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        for template in templates1:
            crop_template = cv2.resize(template, (w, h))

            result = cv2.matchTemplate(crop_img_gray, crop_template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            list_value.append(max_val)

        max_value = max(list_value)
        text = ""
        if max_value > 300000:
            text = texts1[list_value.index(max_value)]

        hasblack = False
        if text == "noleft" or text == "noright":
            for row in center_crop_img:
                for col in row:
                    if col[0] < 36 and col[1] < 30 and col[2] < 30:
                        hasblack = True
            if not hasblack:
                text = "stop"

        haswhite = False
        if text == "stop":
            for row in center_crop_img2:
                for col in row:
                    if col[0] > 130 and col[1] > 130 and col[2] > 130:
                        haswhite = True
            if not haswhite:
                text = ""

        list_text.append(text)

        if draw is not None and text != "":
            cv2.rectangle(draw, (x, y), (x + w, y + h), (8, 255, 255), 3)
            cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 8, 255), 2)

    # BLUE SIGNS
    for bbox in bboxes_b:
        x, y, w, h = bbox

        list_value = []
        crop_img = img[y:y + h, x:x + w]
        center_crop_img = crop_img[0:int(h * 0.15), 0:w]
        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        for template in templates2:
            crop_template = cv2.resize(template, (w, h))

            result = cv2.matchTemplate(crop_img_gray, crop_template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            list_value.append(max_val)

        max_value = max(list_value)
        text = ""
        if max_value > 300000:
            text = texts2[list_value.index(max_value)]

        haswhite = False
        if text == "left" or text == "right":
            for row in center_crop_img:
                for col in row:
                    if col[0] > 180 and col[1] > 180 and col[2] > 180:
                        haswhite = True
            if haswhite:
                text = "straight"

        list_text.append(text)

        if draw is not None and text != "":
            cv2.rectangle(draw, (x, y), (x + w, y + h), (8, 255, 255), 3)
            cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 8, 255), 2)

    return list_text


def calculate_speed_angle(signs, top_point):
    global turnleft_count, turnright_count, turnleft, turnright, gostraight_count, gostraight, duration, speed, steer, stop

    speed = 0.5
    img_centerx = IMAGE_W // 2

    center_point_right = center + lane_width / 4
    center_point_left = center - lane_width / 4

    center_point = center
    if has_car_left and not has_car_right:
        center_point = center_point_right
    if has_car_right and not has_car_left:
        center_point = center_point_left

    center_diff = img_centerx - center_point

    steer = - float(center_diff * 0.033)

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
        speed = 0
        steer = -1
        if turnleft_count >= duration:
            turnleft_count = 0
            duration = 0

    if turnright_count > 0:
        turnright_count += 1
        speed = 0
        steer = 1
        if turnright_count >= duration:
            turnright_count = 0
            duration = 0

    if gostraight_count > 0:
        gostraight_count += 1
        steer = 0
        if gostraight_count >= duration:
            gostraight_count = 0
            duration = 0

    speed -= abs(steer) * 1.3
    if 0.85 * IMAGE_H - top_point <= lane_width * 1.3:
        speed = 0.1
    if speed <= 0:
        speed = 0.1

    if stop > 0:
        stop += 1
        speed = 0
        if stop >= duration:
            stop = 0
            duration = 0

    return speed, steer

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

    if not gs.rgb_frames.empty():
        rgb = gs.rgb_frames.get()

    if not gs.depth_frames.empty():
        depth = gs.depth_frames.get()

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    IMAGE_H, IMAGE_W = image.shape[:2]

    # Birdview transformation and Canny
    image_birdview = birdview(image)
    canny_image = canny(image_birdview)

    # Prepare images to be draw on
    point_image = birdview(image)
    traffic_image = image.copy()
    hough_image = birdview(image)

    # Prepare images for algorithm
    w_filter = white_filter(image_birdview)
    hough = houglines(w_filter, hough_image)
    mask_car = car_filter(image_birdview)

    # Algorithm
    left_right_point(w_filter, hough, mask_car, draw=point_image)
    top_right_left_point(hough, draw=point_image)
    signs = detect_traffic_signs_without_model(image, draw=traffic_image)
    top_point = detect_top_point(mask_car, draw=point_image)

    # Mesure and decide speed and angle
    speed, steer = calculate_speed_angle(signs, top_point)
    
    gs.speed = speed * 50
    gs.steer = steer * 100

    # Show images
    # cv2.imshow("image", image)
    cv2.imshow("canny", canny_image)
    cv2.imshow("white filter", w_filter)
    cv2.imshow("hough", hough_image)
    cv2.imshow("point", point_image)
    cv2.imshow("traffic", traffic_image)
    cv2.imshow('mask_car', mask_car)

    cv2.waitKey(1)
    message = json.dumps({"speed": speed, "steer": steer})
    print(message)

mc.join()
camera.join()
