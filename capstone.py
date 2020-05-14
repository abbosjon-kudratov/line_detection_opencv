import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

'''
This code has been written on 14.05.2020
by team53 for Capstone Design SOC4150
Final project @ Inha University in Tashkent

written by Abbosjon Kudratov

follow me on twitter:
twitter.com/a_kudratov

'''

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def morph_closing(image):
    # P.S. erosion and dilation  => ONLY APPLIED TO GRAY IMAGES!
    # create 5x5 kernel with 1s
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    kernel3 = np.ones((9, 9), np.uint8)
    kernel4 = np.array([[0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0]], dtype=np.uint8)

    # apply  dilation
    #morphological closing = dilation , then erosion
    frame_dilation = cv2.dilate(image, kernel2, iterations=2)
    frame_erosion = cv2.erode(frame_dilation, kernel2, iterations=2)

    #or alternatively
    closing = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel2,iterations=2)

    cv2.imshow('morpholofical closing effect',closing)


    return closing


def ROI_area_masking(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = (255,)  # * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)

    # we can use cv2.bitwise_and() to do masking:
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # for x1, y1, x2, y2 in line:
            try:
                cv2.line(blank_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), thickness=5)
            except OverflowError:
                pass

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img


def make_coordinates(img, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0

    # slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 4.3))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def avg_slope_intercept(img, lines):
    left_points = []
    right_points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            # we can differentiate lines whether on right or lefft by their slope values:
            if slope < 0:  # points on the left have negative slope
                left_points.append((slope, intercept))  # so we add these points to our left_points array
            else:
                right_points.append((slope, intercept))  # else these points are on the right side

            # to draw the continuous line we have to find averages of these left or right points arrays
            left_points_avg = np.average(left_points,
                                         axis=0)  # axis should be 0 because we want averages of columns in array
            right_points_avg = np.average(right_points, axis=0)

            # print(right_points_avg,'right avg')
            # print(left_points_avg,'left avg')

            # we need the coordinates to draw the line:
            right_line = make_coordinates(img, right_points_avg)
            left_line = make_coordinates(img, left_points_avg)

            return np.array([left_line, right_line])


def process_image(img):
    # resize the frame image:
    if img is not None:
        h, w, d = img.shape
        frame = cv2.resize(img, (int(w / 1.5), int(h / 1.5)))  # store the resized image in another variable

        cv2.imshow('original frame (resized)',frame)
        lane_img = np.copy(frame)
        # apply Gaussian filter to smooth the image:
        blurred = cv2.GaussianBlur(lane_img, (5, 5), 1)

        # cv2.imshow('blurred frame', blurred)

        # convert to grascale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)

        # this code determines the threshold automatically from the image using Otsu's method:
        # and store the binarized image in variable  im_bw
        (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imshow('binarized (black and white)', im_bw)
        # apply canny edge detection with binarized image
        # canny = cv2.Canny(im_bw, 50, 150, apertureSize=3)
        canny = auto_canny(im_bw)

        cv2.imshow('canny edge', canny)

        # apply canny edge detection with only grayscale image
        # canny2 = cv2.Canny(im_bw, 50, 150, apertureSize=3)

        h, w = canny.shape
        # crop_img = canny[int(h // 1.5):h, 0:w]

        ROI_frame = [
            # (0, h),
            # (0, int(h / 2)),
            # (w, int(h / 2)),
            # (int(w/2), int(h/2))
            # (w, h),
            # 350,500  0,500

            (200,500), (800,500),(450,300)
            # (460, 350),
            # (300, 520),
            # (900, h)
            # (500,350)
        ]

        cropped_masked_img = ROI_area_masking(canny, np.array([ROI_frame], np.int32))
        cv2.imshow('masked cropped edge detected', cropped_masked_img)

        cropped_masked_img = morph_closing(cropped_masked_img)



        # Hough Space : images -> lines converted from x,y space to polar mc space (Hough Space)
        lines = cv2.HoughLinesP(cropped_masked_img, rho=4, theta=np.pi / 180,
                                threshold=3, minLineLength=130, maxLineGap=40)

        averaged_lines = avg_slope_intercept(lane_img, lines)

        image_with_lines = draw_the_lines(lane_img, averaged_lines)



        return image_with_lines

    # Here we tried to use HoughLines() function which take more computational power,
    # so we just commented it out for possible future references
    # for line in lines:
    #   x1,y1,x2,y2 = line[0]
    #   cv2.line(blurred, (x1,y1), (x2,y2), (0,255,0),2)

    # lines = cv2.HoughLines(canny, 1, np.pi / 180, 150)
    #
    # if lines is not None:
    #   for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #
    #     # x1 stores rounded off value of (r*cos(theta)-1000*sin(theta))
    #     x1 = int(x0+1000*(-b))
    #
    #     # y1 stores rounded off value of (r*sin(theta)+1000*cos(theta))
    #     y1 = int(y0 + 1000 * (a))
    #
    #     # x2 stores rounded off value of (r*cos(theta)-1000*sin(theta))
    #     x2 = int(x0 - 1000 * (-b))
    #
    #     # y1 stores rounded off value of (r*sin(theta)+1000*cos(theta))
    #     y2 = int(y0 - 1000 * (a))
    #
    #     cv2.line(blurred, (x1,y1), (x2,y2), (0,255,0),2)
    #


# open the video file
videofile = cv2.VideoCapture('./road.mp4')

while True:
    # ret = frame capture result
    # frame = captured frame
    ret, frame = videofile.read()
    frame = process_image(frame)
    try:
        # cv2.imshow("video", frame)
        cv2.imshow('with lines', frame)
    except:
        pass

    # plt.imshow(frame)
    # plt.show()

    if ret is False:
        print('end of video!!: break!')
        break

    q = cv2.waitKey(1) & 0xff
    if q == ord('q'):
        print('q is pressed: quit')
        break

# close the windows
videofile.release()
cv2.destroyAllWindows()
