# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# n iterations
N = 20
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

R_Line = Line()
L_Line = Line()
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# load saved objpoints and imgpoints
objpoints = np.load('output_images/objpoints_save.npy')
imgpoints = np.load('output_images/imgpoints_save.npy')

# load saved objpoints and imgpoints
src = np.load('output_images/warp_src_save.npy')
dst = np.load('output_images/warp_dst_save.npy')
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def cal_undistort(img, objpoints, imgpoints):
    h,w = img.shape[:2]
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    undistorted = cal_undistort(img, objpoints, imgpoints)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    warped = cv2.warpPerspective(color_binary, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    return gray

def find_lane_pixels(binary_warped, DIR):

	# if detection is founded at previous time, no need to find lane pixels
    if DIR.detected == True:
		search_around_poly(binary_warped, DIR)
		return 0
		
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    #plt.plot(histogram)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint    
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
	if DIR == L_Line:
		DIR.allx = nonzerox[left_lane_inds]
		DIR.ally = nonzeroy[left_lane_inds]
	else:
		DIR.allx = nonzerox[right_lane_inds]
		DIR.ally = nonzeroy[right_lane_inds]
	
def search_around_poly(binary_warped, DIR):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    lane_inds = ((nonzerox > (DIR.best_fit[0]*(nonzeroy**2) + DIR.best_fit[1]*nonzeroy + 
                    DIR.best_fit[2] - margin)) & (nonzerox < (DIR.best_fit[0]*(nonzeroy**2) + 
                    DIR.best_fit[1]*nonzeroy + DIR.best_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    DIR.allx = nonzerox[left_lane_inds]
    DIR.ally = nonzeroy[left_lane_inds] 

def fit_polynomial(DIR):

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    line_fit = np.polyfit(DIR.allx, DIR.ally, 2)
    
	if DIR.current_fit.shape[0] <N:
		DIR.current_fit = np.append(DIR.current_fit, line_fit, axis=0)
	else :
		DIR.current_fit = np.append(DIR.current_fit[1:], line_fit, axis=0)

	DIR.diffs = abs(DIR.current_fit[DIR.current_fit.shape[0]-1,:]-right_fit)
	DIR.best_fit = DIR.current_fit.average(axis=0)
        
    fitx_bottom = DIR.best_fit[0]*h**2 + DIR.best_fit[1]*h + DIR.best_fit[2]
	if DIR == L_Line:
		DIR.line_base_pos = w/2-fitx_bottom
	else:
		DIR.line_base_pos = fitx_bottom-w/2

    ##### TO-DO: Implement the calculation of R_curve (meter of curvature) #####
    line_fit_m = np.polyfit(DIR.ally*ym_per_pix, DIR.allx*xm_per_pix, 2)
    curve_m = ((1 + (2*line_fit_m[0]*y_eval*ym_per_pix + line_fit_m[1])**2)**1.5) / np.absolute(2*line_fit_m[0]) ## Implement the calculation of the left line here
    
    #print(format(left_curve_m, ".2f"), 'm', format(right_curve_m, ".2f"), 'm')
    DIR.radius_of_curvature = curve_m

##############################################################################################################
    left_slope_m = (np.polyfit(L_Line.ally*ym_per_pix, L_Line.allx*xm_per_pix, 1))[0]
    right_slope_m = (np.polyfit(R_Line.ally*ym_per_pix, R_Line.allx*xm_per_pix, 1))[0]
    

    drivingLine.detected = sanity_check(left_slope_m, right_slope_m)
    if drivingLine.detected == False:
        cv2.imwrite('failed_images/warped_binary_test1.jpg', result)
    #print(drivingLine.detected)
##############################################################################################################


def draw_line(binary_warped)
	
	# Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	# Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        L_Line.bestx = L_Line.best_fit[0]*ploty**2 + L_Line.best_fit[1]*ploty + L_Line.best_fit[2]
		R_Line.bestx = R_Line.best_fit[0]*ploty**2 + R_Line.best_fit[1]*ploty + R_Line.best_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        L_Line.bestx = 1*ploty**2 + 1*ploty
		R_Line.bestx = 1*ploty**2 + 1*ploty
	
	# Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
	
	# Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([L_Line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([R_Line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    return newwarp
    
	
def sanity_check(left_slope, right_slope):
	proper_distance = False
	if L_Line.line_base_pos<0:
		print("ERR: left line is detected right half-side of image")
	else if R_Line.line_base_pos<0:
		print("ERR: right line is detected left half-side of image")
	else
		line_width = xm_per_pix*(L_Line.line_base_pos+R_Line.line_base_pos)
		if 2 < line_width and line_width < 4:
			proper_distance = True
		else:
			print("ERR: sanity test failed !!! due to unproper lines distance")
    
    if abs(L_Line.radius_of_curvature-R_Line.radius_of_curvature) < 10000:
        similar_curves = True
    else:
        similar_curves = False
        print("sanity test failed !!! due to big difference between curvatures", left_curve, right_curve)
    
    # diff between sople of two lines less then around 20 degrees
    if abs(left_slope-right_slope) < 0.4:
        similar_slpoe = True
    else:
        similar_slpoe = False
        print("sanity test failed !!! due to big difference between slopes")
        
    return (proper_distance and similar_curves and similar_slpoe)

def process_image(image):
    colored_image = np.copy(image)
    result = pipeline(image, (170, 200), (15, 100))
	
	find_lane_pixels(result, L_Line)
	find_lane_pixels(result, R_Line)
	
    fit_polynomial(L_Line)
	fit_polynomial(R_Line)
	
	linedwarp = draw_line(result)
	
    # Combine the result with the original image and put text on image
    out_img = cv2.addWeighted(colored_image, 1, linedwarp, 0.3, 0)
    disp = "Radius of Curvature = " + str(int((R_Line.radius_of_curvature+L_Line.radius_of_curvature)/2)) + "(m)"
    cv2.putText(out_img, disp, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    if L_Line.line_base_pos>R_Line.line_base_pos :
        dir_str = "left"
    else :
        dir_str = "right"
    disp = "Vehicle is " + str(format(abs(img.shape[1]-(L_Line.line_base_pos+R_Line.line_base_pos)/2), ".2f")) + "m " + dir_str + " of center"
    cv2.putText(out_img, disp, (80,135), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    #plt.imshow(result)
    return out_img

white_output = 'project_video_test.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4").subclip(0,0.1)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)