# pipeline for lane detection solidWhiteRight.mp4 & solidYellowLeft.mp4

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from moviepy.editor import *


def img_pipeline_seg(image):
	# gray scale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# Gaussian smoothing
	kernel_size = 5
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
	threshold = 10    # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 10 #minimum number of pixels making up a line
	max_line_gap = 10    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on
	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)
	# Iterate over the output "lines" and draw lines on a blank image
	for line in lines:
	    for x1,y1,x2,y2 in line:
	    	# delete wrong oriented lines by checking k
	    	if x2 == x1:
	    		cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
	    	else:
		    	k = float((y2-y1)/(x2-x1))
		    	if k > 0.4 or k < -0.4:
		 	       cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)

	# Draw the lines on the original image
	lines_img = cv2.addWeighted(image, 0.8, line_image, 1, 0)

	# plt.imshow(lines_img)
	return lines_img

def img_pipeline_line(image):
	# gray scale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# Gaussian smoothing
	kernel_size = 5
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
	min_line_length = 50 #minimum number of pixels making up a line
	max_line_gap = 100    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on
	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)
	# Iterate over the output "lines" and draw lines on a blank image
	for line in lines:
	    for x1,y1,x2,y2 in line:
	    	# delete wrong oriented lines by checking k
	    	if x2 == x1:
	    		cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
	    	else:
		    	k = float((y2-y1)/(x2-x1))
		    	if k > 0.4 or k < -0.4:
		 	       cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

	# Draw the lines on the original image
	lines_img = cv2.addWeighted(image, 0.8, line_image, 1, 0)

	# plt.imshow(lines_img)
	return lines_img


if __name__=='__main__':
	white_seg = "white_seg.mp4"
	clip1 = VideoFileClip("solidWhiteRight.mp4")
	sub_clip = clip1.fl_image(img_pipeline_seg)
	sub_clip.write_videofile(white_seg, audio=False)

	white_line = "white_line.mp4"
	clip1 = VideoFileClip("solidWhiteRight.mp4")
	sub_clip = clip1.fl_image(img_pipeline_line)
	sub_clip.write_videofile(white_line, audio=False)

	yelloe_seg = "yelloe_seg.mp4"
	clip1 = VideoFileClip("solidYellowLeft.mp4")
	sub_clip = clip1.fl_image(img_pipeline_seg)
	sub_clip.write_videofile(yelloe_seg, audio=False)

	yelloe_line = "yelloe_line.mp4"
	clip1 = VideoFileClip("solidYellowLeft.mp4")
	sub_clip = clip1.fl_image(img_pipeline_line)
	sub_clip.write_videofile(yelloe_line, audio=False)