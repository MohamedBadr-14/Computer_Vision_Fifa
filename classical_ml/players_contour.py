import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_players_contours(img):
    height, width = img.shape[:2]

    # Crop 15% of the height from the top
    crop_height = int(height * 0.15)
    cropped_img = img[crop_height:, :, :].copy()  # Crop directly, avoid modifying top

    # Define color bounds for non-green mask
    lower_bound = np.array([0, 0, 0], dtype=np.uint8)
    upper_bound = np.array([179, 80, 255], dtype=np.uint8)

    # Convert to HSV once
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Create non-green mask
    mask_non_green = cv2.inRange(hsv, lower_bound, upper_bound)
    lines_mask = cv2.bitwise_not(mask_non_green)

    # Green detection
    lower_green = np.array([40, 50, 50], dtype=np.uint8)
    upper_green = np.array([54, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks
    binary_img = cv2.bitwise_not(mask_green)
    binary_img = cv2.bitwise_and(binary_img, lines_mask)
    binary_img = (binary_img > 0).astype(np.uint8)

    # Morphological operation
    kernel = np.ones((6, 2), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=2)

    ## draw black mask on
    x1, y1 = 555, 1390
    x2, y2 = 590, 1425
    cv2.rectangle(binary_img, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Fill with black
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cut = []
    correct_contours = []

    # Process contours
    image_cropped_new = cropped_img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # if the contour is in this dimensions then skip
        # x between 540 and 600  
        # y between 1365 and 1445
        # if the contour is in this dimensions then skip
        if 1000 < area < 20000 and h >= w * 0.25:  # Filter small contours
            correct_contours.append(contour)
            contours_cut.append(cropped_img[y:y + h, x:x + w])

    return image_cropped_new, contours_cut, correct_contours 
         

def decide_contours(contour_img):
    # Convert to HSV once
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)

    # Blue detection
    lower_blue = np.array([100, 100, 40], dtype=np.uint8)
    upper_blue = np.array([130, 255, 200], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    binary_img_blue = cv2.bitwise_not(mask_blue)
    binary_img_blue = (binary_img_blue > 0).astype(np.uint8)
    pixels_blue = np.count_nonzero(binary_img_blue == 0)

    # Yellow detection
    lower_yellow = np.array([20, 50, 50], dtype=np.uint8)
    upper_yellow = np.array([35, 180, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    binary_img_yellow = cv2.bitwise_not(mask_yellow)
    binary_img_yellow = (binary_img_yellow > 0).astype(np.uint8)
    pixels_yellow = np.count_nonzero(binary_img_yellow == 0)

    delta = pixels_blue - pixels_yellow

    if pixels_blue > 400 and pixels_yellow > 400 and (delta < -700 or delta > 700):
        # Process yellow contour
        binary_img_yellow = (binary_img_yellow == 0).astype(np.uint8)
        kernel = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((6, 2), np.uint8)
        binary_img = cv2.erode(binary_img_yellow, kernel, iterations=1)
        binary_img = cv2.dilate(binary_img, kernel2, iterations=2)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_yellow = None
        if contours:
            contour_yellow = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            x, y, w, h = cv2.boundingRect(contour_yellow)
            x, y = x - 7, y - 15

        # Process blue contour
        binary_img_blue = (binary_img_blue == 0).astype(np.uint8)
        binary_img = cv2.dilate(binary_img_blue, kernel2, iterations=2)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_blue = None
        if contours:
            contour_blue = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            x, y, w, h = cv2.boundingRect(contour_blue)
            x, y = x - 7, y - 15
            cv2.rectangle(contour_img, (x, y), (x + w + 10, y + h + 70), (0, 255, 0), 2)

        return 2, contour_blue, contour_yellow

    if pixels_yellow > pixels_blue and delta < 0:
        return 1, None, None
    elif pixels_blue > 90:
        return 0, None, None

    return 3, None, None