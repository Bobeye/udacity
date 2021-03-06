import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
# from moviepy.editor import *

def process_image(image):
    # gray scale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Gaussian smoothing
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Canny edge
    low_threshold = 70
    high_threshold = 210
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # polygon region mask
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    imshape = image.shape
    vertices = np.array([[(20,539),(400, 320), (500, 320), (929,539)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    # Hough Transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/720 # angular resolution in radians of the Hough grid
    threshold = 100    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 35 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    # left_ymin = 549 
    # left_xmin = 0
    # right_ymin = 549
    # right_xmin = 0
    # left_k = 0
    # left_nseg = 0
    # right_k = 0
    # right_nseg = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            # delete wrong oriented lines by checking k
            if x2 == x1:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            else:
                k = float((y2-y1)/(x2-x1))
                if k > 0.4 or k < -0.4:
                    b = x1-(1/k)*y1
                    if y1 < y2:
                        y2 = 539
                        x2 = int((1/k)*y2+b)
                    else:
                        y1 = 539
                        x1 = int((1/k)*y1+b)

                    print (k, x1, y1, x2, y2)
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    # left_k = left_k / left_nseg
    # left_b = left_xmin-(1/left_k)*left_ymin
    # left_ymax = 539
    # left_xmax = int((1/left_k)*left_ymax + left_b)
    # print (left_ymin, left_xmin, left_ymax , left_xmax, left_k/left_nseg)

    # Draw the lines on the original image
    lines_img = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # plt.imshow(lines_img)
    plt.imshow(lines_img)
    result = lines_img

    return result



if __name__=='__main__':
    files = glob.glob('test_images/'+'*.jpg')
    image = mpimg.imread(files[3])
    process_image(image)
    plt.show()
