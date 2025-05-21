import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_frames(frames_path):
        # Read the frames from the specified path
        frames = []
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                # frame = cv2.resize(frame, (640, 640)) 
                # if count == 12:
                frames.append(frame)
      
                if count == 70:
                    break
            
                count += 1
        return frames

def get_players_contours(img):

    height, width = img.shape[:2]

    # Crop 20% of the height from the top
    crop_height = int(height * 0.15)
    # crop_width = int(width * 0.1)
    cropped_img = img.copy()
    cropped_img[0:crop_height] = 0   
    # display the cropped image


# Your target BGR color
    bgr_color = np.uint8([[[181, 187, 173]]])  # BGR format
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    # print("HSV color:", hsv_color)  # Just to check

    # Define a tolerance range
    h, s, v = hsv_color
    lower_bound = np.array([0,0, 0])  # Lower bound for hue, saturation, and value
    upper_bound = np.array([179, 80, 255])  # Upper bound for hue, saturation, and value

    # Convert input image to HSV
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Create the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Invert the mask
    lines_mask = cv2.bitwise_not(mask)

    # plt.imshow(lines_mask, cmap='gray')
    # plt.title("Green=0, Others=1 (Binary Image)")
    # plt.axis('off')
    # # plt.show()







  
    
  







    # Convert to HSV color space for better green detection
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)



    lower_green = np.array([40, 50, 50])  # Hue: 35-80 covers green shades
    upper_green = np.array([54, 255, 255])


    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert the mask: green areas (0), others (255)
    binary_img = cv2.bitwise_not(mask)

    # Convert to binary: green=0, others=1
    binary_img = (binary_img > 0).astype(np.uint8)
    # pefform bit wise and 
    binary_img = cv2.bitwise_and(binary_img, lines_mask)

    # Display the binary image
    plt.imshow(binary_img, cmap='gray')
    plt.title("Green=0, Others=1 (Binary Image)")
    plt.axis('off')
    # plt.show()

    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, threshold=100, minLineLength=70, maxLineGap=10)

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         # Check if the line length is greater than 10% of the image width
    #         line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #         print(f"Line length: {line_length}")
    #         if line_length >95:
    #             cv2.line(binary_img, (x1, y1), (x2, y2), (0, 0, 0), 2)


    # kernel = np.ones((4, 4), np.uint8)
    kernel2 = np.ones((6, 2), np.uint8)
    # binary_img = cv2.erode(binary_img, kernel, iterations=1)     
    binary_img = cv2.dilate(binary_img, kernel2, iterations=2)  # Dilate to restore the original size       

    # plt.imshow(binary_img, cmap='gray')
    # plt.title("Green=0, Others=1 (Binary Image)")
    # plt.axis('off')
    # # plt.show()
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cut = []
    correct_contours = []
    # Draw contours on the original image rectangle
    image_cropped_new=cropped_img.copy()
    for contour in contours:
        # print("Contour area:", cv2.contourArea(contour))
        area = cv2.contourArea(contour)
        width = cv2.boundingRect(contour)[2]
        height = cv2.boundingRect(contour)[3]
        if  (area> 1000 and area<20000) and not (height<1/4*(width) ):  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            correct_contours.append(contour)
            cropped_contour = cropped_img[y:y + h, x:x + w]
            contours_cut.append(cropped_contour)
            # cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # put on array frames the cropped part of contour

    # plt.imshow(cropped_img)
    # plt.title("Green=0, Others=1 (Binary Image)")
    # plt.axis('off')
    # plt.show()  
    # i want to save the image with the contours in folder output

    return image_cropped_new, contours_cut, correct_contours
  


         
def decide_contours(contour_img):
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 40])  # Hue: 35-80 covers green shades
    upper_blue = np.array([130, 255, 200])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    binary_img = cv2.bitwise_not(mask)
    binary_img_blue = (binary_img > 0).astype(np.uint8)    # pefform bit wise and 
    pixels_blue=0
    for x in range(binary_img_blue.shape[0]):
        for y in range(binary_img_blue.shape[1]):
            if binary_img_blue[x,y]==0:
                pixels_blue+=1
    pixels_yellow=0
    hsv = cv2.cvtColor(contour_img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 50, 50])  # Hue: 35-80 covers green shades
    upper_yellow = np.array([35, 180, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    binary_img_new = cv2.bitwise_not(mask)
    binary_img_yellow = (binary_img_new > 0).astype(np.uint8)    # pefform bit wise and 
    for x in range(binary_img_yellow.shape[0]):
        for y in range(binary_img_yellow.shape[1]):
            if binary_img_yellow[x,y]==0:
                pixels_yellow+=1

        # print("Pixels blue:", pixels_blue)
        # print("Pixels yellow:", pixels_yellow)
    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(binary_img_blue, cmap='gray')
    # plt.title("Blue Mask")
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(binary_img_yellow, cmap='gray')
    # plt.title("Yellow Mask")
    # plt.axis('off')

    # plt.tight_layout()
    # # plt.show()

    delta=pixels_blue-pixels_yellow
    # print("Delta:", delta)
    if pixels_blue>400 and pixels_yellow>400 and (delta<700 or delta<-700):
        binary_img_yellow=(binary_img_yellow==0).astype(np.uint8)
        kernel = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((6, 2), np.uint8)
        # binary_img = cv2.erode(binary_img, kernel, iterations=1) 
        binary_img = cv2.erode(binary_img_yellow, kernel, iterations=1)     
        binary_img = cv2.dilate(binary_img, kernel2, iterations=2)  # Dilate to restore the original size       

        # plt.imshow(binary_img, cmap='gray')
        # plt.title("Green=0, Others=1 (Binary Image)")
        # plt.axis('off')
        # plt.show()
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_yellow=0
        if len(contours)>0:
            contour_yellow= sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]

        
            area = cv2.contourArea(contour_yellow)
        
            # if  (area> 1000 and area<20000) and not (height<1/4*(width)):  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour_yellow)
            x=x-7
            y=y-15
        
            # cv2.rectangle(contour_img, (x, y), (x + w+10, y + h+70), (0, 255, 0), 2)

            # plt.imshow(contour_img)
            # plt.title("Green=0, Others=1 (Binary Image)")
            # plt.axis('off')
            # plt.show()



        binary_img_blue=(binary_img_blue==0).astype(np.uint8)
        binary_img = cv2.dilate(binary_img_blue, kernel2, iterations=2)  # Dilate to restore the original size       

        # plt.imshow(binary_img, cmap='gray')
        # plt.title("Green=0, Others=1 (Binary Image)")
        # plt.axis('off')
        # plt.show()
        # # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_blue=0
        if len(contours)>0:
        # i want to sort contours to take the biggest one area
            contour_blue= sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]

        
            area = cv2.contourArea(contour_blue)
        
            # if  (area> 1000 and area<20000) and not (height<1/4*(width)):  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour_blue)
            x=x-7
            y=y-15
        
            cv2.rectangle(contour_img, (x, y), (x + w+10, y + h+70), (0, 255, 0), 2)

        # plt.imshow(contour_img)
        # plt.title("Green=0, Others=1 (Binary Image)")
        # plt.axis('off')
        # plt.show()
            
            








        return 2,contour_blue,contour_yellow    


    if pixels_yellow>pixels_blue and delta<0:
    
        return 1,None,None
    elif pixels_blue>90:
        
        return 0,None,None
    
    return 3,None,None
      



     
        
                    



 

    # Save the output



# frames = read_frames('all')
# output_dir = "new_contours"
# os.makedirs(output_dir, exist_ok=True)

# for idx, frame in enumerate(frames):
#     output_image,contours = get_players_contours(frame)
#     output_path = os.path.join(output_dir, f"frame_{idx+1}.png")
#     for i, contour in enumerate(contours):
#         # Save each contour as a separate image
#         contour_path = os.path.join(output_dir, f"contour_{idx+1}_{i+1}.png")
#         cv2.imwrite(contour_path, contour)


# img= cv2.imread('coll_1.png')
# decide_contours(img)    















# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(img)
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(binary_img_blue, cmap='gray')
# plt.title("Blue Mask")
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(binary_img_yellow, cmap='gray')
# plt.title("Yellow Mask")
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# # Get image dimensions
# crop_and_segment_green_field(img)