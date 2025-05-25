import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import itertools

def get_players_contours(img):
    height, width = img.shape[:2]
    print('image shape', img.shape)
    # Crop 15% of the height from the top
    crop_height = int(height * 0.15)
    cropped_img = img[crop_height:, :, :].copy()  # Crop directly, avoid modifying top
    # print('cropped image shape', cropped_img.shape)
    # Define color bounds for non-green mask
    lower_bound = np.array([0, 0, 110], dtype=np.uint8)
    upper_bound = np.array([180, 50, 160], dtype=np.uint8)

    # Convert to HSV once
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # try_white_ranges(hsv)
    # try_green_ranges(hsv)
    # Create non-green mask
    mask_non_green = cv2.inRange(hsv, lower_bound, upper_bound)
    # lines_mask = cv2.dilate(mask_non_green, np.ones((3, 3), np.uint8), iterations=1)
    lines_mask = cv2.bitwise_not(mask_non_green)
    # cv2.imshow('lines_mask', lines_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Get stats from hsv where lines_mask == 1
    # mask_indices = lines_mask > 0
    # hsv_lines = hsv[mask_indices]
    # if hsv_lines.size > 0:
    #     h_vals = hsv_lines[:, 0]
    #     s_vals = hsv_lines[:, 1]
    #     v_vals = hsv_lines[:, 2]
    #     print("Lines mask HSV stats:")
    #     print(f"  H: mean={np.mean(h_vals):.2f}, std={np.std(h_vals):.2f}, min={np.min(h_vals)}, max={np.max(h_vals)}")
    #     print(f"  S: mean={np.mean(s_vals):.2f}, std={np.std(s_vals):.2f}, min={np.min(s_vals)}, max={np.max(s_vals)}")
    #     print(f"  V: mean={np.mean(v_vals):.2f}, std={np.std(v_vals):.2f}, min={np.min(v_vals)}, max={np.max(v_vals)}")
    # else:
    #     print("No pixels found in lines_mask for HSV stats.")
    # # Print most repeated H, S, and V values
    # try_black_ranges(hsv)
    lower_black = np.array([10, 62, 30], dtype=np.uint8)
    upper_black = np.array([40, 142, 70], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Green detection
    lower_green = np.array([30, 60, 60], dtype=np.uint8)
    upper_green = np.array([60, 140, 200], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # cv2.imshow('mask_green', mask_green)
    # cv2.waitKey(0)

    # Combine masks
    mask_black = cv2.bitwise_not(mask_black)
    binary_img = cv2.bitwise_not(mask_green)
    # cv2.imshow('binary_img 1', binary_img)
    # cv2.waitKey(0)
    binary_img = cv2.bitwise_and(binary_img, lines_mask)
    # cv2.imshow('binary_img 2', binary_img)
    # cv2.waitKey(0)
    # cv2.imshow('mask_black', mask_black)
    # cv2.waitKey(0)
    binary_img = cv2.bitwise_and(binary_img, mask_black)
    # cv2.imshow('binary_img 2', binary_img)
    # cv2.waitKey(0)
    # cv2.imshow('lines mask1', lines_mask)
    # cv2.waitKey(0)
    binary_img = (binary_img > 0).astype(np.uint8)*255
    # cv2.imshow('binary_img3', binary_img)
    # cv2.waitKey(0)

    # Morphological operation
    # kernel = np.ones((6, 2), np.uint8)
    # binary_img = cv2.dilate(binary_img, kernel, iterations=2)

    # cv2.imshow('after dilation', binary_img)
    # cv2.waitKey(0)
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cut = []
    correct_contours = []

    # Process contours
    image_cropped_new = cropped_img.copy()
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    i=0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # if the contour is in this dimensions then skip
        # x between 540 and 600  
        # y between 1365 and 1445
        # if the contour is in this dimensions then skip
        # print('contour area', area, 'x', x, 'y', y, 'w', w, 'h', h)
        # cv2.imshow('contour', cropped_img[y:y + h, x:x + w])
        # cv2.waitKey(0)
        if 400 < area < 20000 and h >= w * 0.25:  # Filter small contours
            correct_contours.append(contour)
            contours_cut.append(cropped_img[y:y + h, x:x + w])
            # if not os.path.exists('dataset'):
            #     os.makedirs('dataset')
            # filename = f'dataset/player2{i}_{x}_{y}_{w}_{h}.png'
            # cv2.imwrite(filename, cropped_img[y:y + h, x:x + w])
            # i+=1
    # print('contours_cut', len(contours_cut))
    # Draw contours on the image
    for contour in correct_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_cropped_new, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_cropped_new, str(len(correct_contours)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Show the image with contours

    # cv2.imshow('Contours', image_cropped_new)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_cropped_new, contours_cut, correct_contours 
         

def decide_contours(contour_img,frame,x,y):
    # Convert to HSV once
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
    # Print histograms for H, S, and V channels
   

    # orange detection
    # try_orange_ranges(hsv)
    lower_orange = np.array([5,150,100], dtype=np.uint8)
    upper_orange = np.array([19,255,240], dtype=np.uint8)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # cv2.imshow('real', contour_img)
    # cv2.waitKey(0)
    # cv2.imshow('mask_orange', mask_orange)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    binary_img_orange = cv2.bitwise_not(mask_orange)
    # hsv[binary_img_orange == 0] = 0
    # cv2.imshow('hsv', hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # h, s, v = cv2.split(hsv)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.hist(h.ravel(), bins=range(0, 181, 5), color='r', alpha=0.7)
    # plt.title('Hue Histogram')
    # plt.xlabel('Hue')
    # plt.ylabel('Frequency')
    # plt.subplot(1, 3, 2)
    # plt.hist(s.ravel(), bins=range(0, 256, 5), color='g', alpha=0.7)
    # plt.title('Saturation Histogram')
    # plt.xlabel('Saturation')
    # plt.ylabel('Frequency')
    # plt.subplot(1, 3, 3)
    # plt.hist(v.ravel(), bins=range(0, 256, 5), color='b', alpha=0.7)
    # plt.title('Value Histogram')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.show()
    binary_img_orange = (binary_img_orange > 0).astype(np.uint8)
    pixels_orange = np.count_nonzero(binary_img_orange == 0)
    # Convert RGB to HSV
    # rgb_color = np.uint8([[[150,150,130]]])  # RGB format
    # hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    # print("HSV value:", hsv_color[0][0])
    # try_yellow_ranges(hsv)
    # Yellow detection
    lower_yellow = np.array([20, 60, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 240, 240], dtype=np.uint8)
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # cv2.imshow('real', contour_img)
    # cv2.waitKey(0)
    # cv2.imshow('mask_yellow', mask_yellow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    binary_img_yellow = cv2.bitwise_not(mask_yellow)
    binary_img_yellow = (binary_img_yellow > 0).astype(np.uint8)*255
 
   
    pixels_yellow = np.count_nonzero(binary_img_yellow == 0)
    # print('pixels_yellow', pixels_yellow)
    # print('pixels_orange', pixels_orange)
    # cv2.imshow('asly', contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    delta = pixels_orange - pixels_yellow
     # Linearly interpolate the area bounds based on y
            # Fraction of y between y_min and y_max
    y_min = 0  # Top of the image (farthest from camera)
    y_max = frame.shape[0]  # Bottom of the image (closest to camera)
    orange_count_min = 40  # Smallest area when the ball is farthest
    yellow_count_min = 30
    orange_count_max = 200  # Largest area when the ball is closest
    yellow_count_max = 200
    y_fraction = (y+162 - y_min) / (y_max - y_min)
    
    # Interpolate area bounds
    orange_threshold = orange_count_min + y_fraction * (orange_count_max - orange_count_min)
    yellow_threshold = yellow_count_min + y_fraction * (yellow_count_max - yellow_count_min)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    label = f"Y:{pixels_yellow:.2f}, O:{pixels_orange:.2f} YT:{yellow_threshold:.2f}, OT:{orange_threshold:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(frame, label, (x+10, y+162), font, font_scale, (0, 0, 0), thickness)
# 150
    if pixels_orange > orange_threshold-20 and pixels_yellow > yellow_threshold-15:
        # Process yellow contour
        # cv2.imshow('contour_img', contour_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binary_img_yellow = (binary_img_yellow == 0).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        kernel2 = np.ones((7, 3), np.uint8)
        binary_img = cv2.erode(binary_img_yellow, kernel, iterations=1)
        binary_img = cv2.dilate(binary_img, kernel2, iterations=3)
        # cv2.imshow('real', contour_img)
        # cv2.waitKey(0)
        # cv2.imshow('process yellow', binary_img*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_yellow = None
        if contours:
            contour_yellow = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            x, y, w, h = cv2.boundingRect(contour_yellow)
            x, y = x - 5, y - 15
            cv2.rectangle(contour_img, (x, y), (x + w + 5, y + h + 50), (0, 255, 0), 2)
            # cv2.imshow('real', contour_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            

        # Process orange contour
        binary_img_orange = (binary_img_orange == 0).astype(np.uint8)
        binary_img = cv2.erode(binary_img_orange, kernel, iterations=1)
        binary_img = cv2.dilate(binary_img, kernel2, iterations=2)
        # cv2.imshow('contour_img', contour_img)
        # cv2.waitKey(0)
        # cv2.imshow('process orange', binary_img*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_orange = None
        if contours:
            contour_orange = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            x, y, w, h = cv2.boundingRect(contour_orange)
            x, y = x - 5, y - 15
            cv2.rectangle(contour_img, (x, y), (x + w + 5, y + h + 50), (255, 0, 0), 2)
            # cv2.imshow('contour_img', contour_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return 2, contour_orange, contour_yellow

    if pixels_yellow > pixels_orange and pixels_yellow>yellow_threshold:
        return 1, None, None
    elif pixels_orange > pixels_yellow and pixels_orange>orange_threshold:
        return 0, None, None

    return 3, None, None


def try_yellow_ranges(hsv_img):
    # Try ranges around the current yellow: [15, 80, 120] to [40, 240, 240]
    h_lows = range(10, 21, 2)    # Try H from 10 to 20
    h_highs = range(35, 46, 2)   # Try H from 35 to 45
    s_lows = range(60, 101, 10)  # Try S from 60 to 100
    s_highs = range(200, 251, 10) # Try S from 200 to 250
    v_lows = range(100, 141, 10) # Try V from 100 to 140
    v_highs = range(200, 241, 10) # Try V from 200 to 240

    for h_low, h_high in itertools.product(h_lows, h_highs):
        for s_low, s_high in itertools.product(s_lows, s_highs):
            for v_low, v_high in itertools.product(v_lows, v_highs):
                lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
                upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
                mask = cv2.inRange(hsv_img, lower, upper)
                # print(f"Showing mask for lower={lower}, upper={upper}")
                cv2.imshow(f"Mask H:{h_low}-{h_high} S:{s_low}-{s_high} V:{v_low}-{v_high}", mask)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:  # ESC to break early
                    return
                
def try_black_ranges(hsv_img):
    # Typical black: H = any, S = any, V ≈ 0-50 (low value)
    # For RGB (50,50,30):
    # Convert to HSV using OpenCV
    rgb_color = np.uint8([[[50, 50, 30]]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]
    # print("HSV for (50,50,30):", hsv_color)
    # Example: [ 30 102  50] (may vary slightly)
    # Good ranges:
    h_lows = range(max(0, hsv_color[0] - 20), min(180, hsv_color[0] + 21), 10)
    h_highs = range(max(0, hsv_color[0] + 10), min(180, hsv_color[0] + 41), 10)
    s_lows = range(max(0, hsv_color[1] - 40), min(256, hsv_color[1] + 1), 20)
    s_highs = range(hsv_color[1], min(256, hsv_color[1] + 81), 20)
    v_lows = range(max(0, hsv_color[2] - 20), min(256, hsv_color[2] + 1), 10)
    v_highs = range(hsv_color[2], min(256, hsv_color[2] + 41), 10)

    for h_low, h_high in itertools.product(h_lows, h_highs):
        for s_low, s_high in itertools.product(s_lows, s_highs):
            for v_low, v_high in itertools.product(v_lows, v_highs):
                lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
                upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
                mask = cv2.inRange(hsv_img, lower, upper)
                # print(f"Showing mask for lower={lower}, upper={upper}")
                cv2.imshow(f"Mask H:{h_low}-{h_high} S:{s_low}-{s_high} V:{v_low}-{v_high}", mask)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:  # ESC to break early
                    return

# Example usage:
# img = cv2.imread("your_image.jpg")
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# try_yellow_ranges(hsv_img)