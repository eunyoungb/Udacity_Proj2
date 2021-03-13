## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortedChessboard.png "Undistorted"
[image2]: ./output_images/undistortedRawImage.jpg "Road Transformed"
[image3]: ./output_images/thresholded_binary.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/onlySchannelEdgeDetection.jpg "Fit Visual Err"
[image6]: ./output_images/s_channel_withCircle.jpg "S-Channel image"
[image7]: ./output_images/addGrayChannel.jpg "Using Gray Channel image additionally"
[image8]: ./output_images/addGrayScaleEdgeDetection.jpg "Fit Visual"
[image9]: ./output_images/final_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images" at ".Udacity_Proj2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at "3. Use color transforms, gradients, etc., to create a thresholded binary image" at ".Udacity_Proj2.ipynb").  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in "4. Apply a perspective transform to rectify binary image ("birds-eye view")" at ".Udacity_Proj2.ipynb".  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = image.shape[:2]
src = np.float32([[535,453],[30,h],[1250,h],[1280-535,453]])
dst = np.float32([[80,0],[80,h],[1280-80,h],[1280-80,0]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 535, 453      | 80, 0         | 
| 30, 720       | 80, 720       |
| 1250, 720     | 1200, 720     |
| 745, 453      | 1200, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how you detect lane-line pixels more elaborately?

As you can see in above picture, right lane is not detected properly. This is because we did polyfit with 2nd order but detected lanes are only 2 peices. This caused mis-detection continuously. To solve this, I went back to #2.

I checked the S-channel img. You can see, just after we change color to S-channel, many of lines were already missed and could see only 2 pieces of lines. I coucluded like, from here, even if we calibrated elaboratly, line cannot be detected correctly. 

![alt text][image6]

As a solution, I used GrayScale image additionally. The GrayScale img has information which were missed in S-Channel image. So, you can see below table that with additional GrayScale image, we can detect more peices of lines.

![alt text][image7]

Finally, I could polyfit correct line as below. 

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in "7. Determine the curvature of the lane and vehicle position with respect to center & Warp the detected lane boundaries back onto the original image" at ".Udacity_Proj2.ipynb".

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in "7. Determine the curvature of the lane and vehicle position with respect to center & Warp the detected lane boundaries back onto the original image" at ".Udacity_Proj2.ipynb".  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_lineDetected.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Based on proposed approach, almost all lines are detected in each frames. But there is something needs to be updated. 

First of all, the speed of code is late. This lane detection algorithm needs to be running on real-time. But For about 50sec video, rendering took more than 15min. Optimization should be done addtionally to use this algorithm. 

Secondly, when I fix the error case which is introduced in here, I only focused on calibration from color transform. And I gave up calibration and using grayscale image addtionally. This is because I found that even for S-channel img, many information which were existed in raw image was disappeard. But while I am writing this, I got new idea that maybe "perspective transform" could have solved the problem. If I have done more proper "perspective transform", with only 2 peices of lines, polyfit could have done succussfully. This is deserved to try.