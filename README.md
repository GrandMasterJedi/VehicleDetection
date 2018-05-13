[//]: # (Image References)
[image1]: ./example/CarImages.png "car"
[image2]: ./example/noCarImages.png "noCar"
[image3]: ./example/HOGlines.png "hog"
[image4]: ./example/WindowSearchVehicle.png "WinSearch"
[image5]: ./example/WindowSearchPipeline.png "WinPip"
[image6]: ./output_images/test5.jpg "FalsePosFilter"
[video1]: ./carDetectionVideo.mp4 "Video"


# **Vehicle Detection**

Suppose a camera is mounted on the car taking video of the road ahead. I provide a pipeline to detect vehicles from the video in order to support autonomous driving.


---
## Dependencies (Sorted by Dependency Level High to Low)
* Python 3.x
* NumPy
* OpenCV
* scikit-learn
* matplotlib
* MoviePy
* SciPy multi-dimensional Image (ndimage)
* glob, pickle, time, OS 



---
## Remarks
* Vehicle training dataset (folder _resource/image/_) is excluded from this repository. It consists of images of vehicles and non-vehicles used to train the machine learning model and can be downloaded with the links provided in *Reference*
* [_functions.py_](functions.py) consists of function definitions provided during the Udacity nanodegree lecture videos. Two functions are corrected to fit in my current pipeline: `extractFeatures()` [line 102] and `drawLabeledBboxes()` [line 306].


---
## Goals / Steps
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Normalize HOG features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
## Details (Rubric Points)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images

HOG features are extracted using `extractFeatures()` in section 1 of  [_main.ipynb_](main.ipynb). The function, defined in [_functions.py_](functions.py), line 102, calls `get_hog_features()` (line 8) and uses the _Sci-learn Image_ implementation `hog()` .

I apply the HOG feature extraction to all `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]




#### 2. Explain how you settled on your final choice of HOG parameters
The final choice of the `hog()` parameters is based on experimentation and visual inspection with several images. It has not been cross-validated based on the machine learner. 

Specifically, I use the natural _RGB_ color space for the HOG feature extraction because it was proven to relatively robust in a previous exercise.  A few different values for _orientation binning_ and _pixels per cell_ is experimented. The following are the parameters I used to extract hog features in both test images and video.
* Color space: RGB
* Channel: (all)
* Orientations: 30
* Pixels per cell: 16
* Cells per block: 2

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`
and  `RGB` color space and HOG parameters of `orientations=30`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)`

![alt text][image3]

Overall, it seems that higher orientation binning and pixels per cell lead the HOG to focus on macro features of the image. This can potentially reduce false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features

I trained a Support Vector Machine (SVM) to classify the images into *car* and *noCar* using the sklearn implementation `svm.SVC()`. The code can be found in the last part of section 1 of [_main.ipynb_](main.ipynb), in a subsection termed _Apply Feature Extraction to all images (vehicles and non-vehicles)_.

The images are first standardized by using the Scikit-learn's StandardScaler. In an attempt to choose the best learning model, I split the sample in 2/3 for training and 1/3 for testing. I tried several kernels for the SVM and they yield the following performance on the test set:
* Prediction accuracy with *polynomial kernel of order 2*:     	0.934
* Prediction accuracy with *linear kernel* :     				0.953
* Prediction accuracy with *sigmoid kernel*:    				0.784
* Prediction accuracy with *radial basis function kernel*:     	0.974

Therefore, the radial basis function is selected kernel for the SVM.



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is called in section 2 of [_main.ipynb_](main.ipynb). The functions `slide_window()` and `search_windows()` are in respectively line of 198 and line 250 of [_functions.py_](functions.py). 

Based on a window size and windows overlapping parameters, `slide_window` return a list of windows for each image, while `search_windows` search for features in each window that are similar to a car.

First, I define an area of interest on the entire image space. I leave out the higher half section of the image, consisting of mainly sky section, as it is above the picture horizon, and a small horizontal section at the bottom of the image. (`yMinMax = [int(0.5*h), int(0.95*h)]`, where h is the height of the image). Then I run three times sliding window search for each image based on the following strategy for window size and window overlapping parameter:
* Small window search with size (60, 60) and overlapping (0.6, 0.6)
* Medium window search with size (120, 120) and overlapping (0.7, 0.7)
* Large window search with size (200, 200) and overlapping (0.8, 0.8)

The window search based on windows of different size is justified by the fact that cars that are far away appear with smaller dimension, while cars that are closer appear with bigger dimensions.

Below is an example of hot windows, delimited by blue boxes, where the machine learner predicts the presence of car. 
![alt text][image4]

In this example, several false positives and duplicate windows are present.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As a first attempt to reduce false positive and false negative, a heat map is applied on top of detected windows. The intensity of the heat map corresponds to the number of overlapping hot windows. The parameter `heatmapThresh` is a threshold parameter controlling false negatives and is set to 3. If the number of overlapping hot windows is <=3 than it is not considered a window with car, while if the number of overlapping window is higher than 3 than it considered effectively a window classified as a _car_. The result is very sensitive to this threshold parameter. I set it to 3 yielding good compromise between type 1 and type 2 error during experimentation. 

The shape of the resulting window is basically the rectangle that is the overlapping section of 4 or more hot windows. A final filter on the shape of this resulting window is applied as a sanity check to reduce the number of false negative. The following is implemented in `drawLabeledBboxes()`, line 306 of [_functions.py_](functions.py), excluding strangely shaped windows.

```python
    aspectRatio = width / heigth
    bboxArea = width * heigth
	if bboxArea < areaMin or aspectRatio > 3 and aspectRatio < 1/2:
		_NoCarInWindow_
```	

Here is an example of the pipeline, consisting of sliding windows, heat map and sanity check applied to the test image:
![alt text][image5]


In the following example, it is clear that the sanity filter for the shape of the window worked out,
![alt text][image6]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The sliding window car detection technique is applied to a video clip. Check out the video [here](./carDetectionVideo.mp4), or download and open with your computer video reader _carDetectionVideo.mp4_


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As the video is a sequence of images, I combine detected car windows from previous sequences of image in an attempt to further improve the detection accuracy. The last section of [_main.ipynb_](main.ipynb) contains a class `cHotWindows()` that keeps in memory previously found hot windows and a function definition `detectCarImSequence()` that detect car window for a current frame by using information from previous frames. The threshold applied to heatmap of car is set to a multiple of number of frames

```python
	# Set the threshold based on number of frame and number of window identified
    heatmapThresh = int(baseheatmapThresh * nFrame)
```


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The choice of having 3 different sized window for the search works pretty well. However, some further parameter tuning can be done without necessarily overfitting to the current project video. 

The feature extracted for the vehicle classifier can be made more rich, to include different color schemes and/or using deep convolutional neural network with more images, especially for _noCar_. Although the prediction performance on the test dataset is 97%, in my current pipeline landscape elements are sometime detected as car. I suspect that the training dataset does not contain enough _noCar_ features.

Other information can be used to reduce the presence of false positives. For instance, the position of the detected hot window can give an estimate of the distance of the car. Also the size of the window gives such information. This two information can be combined to filter out unreasonable big windows far away in the horizon (or small windows present down on the bottom of the image)

In the current project, portions of the image, possibly corresponding to the sky are excluded, however in downhill or uphill this should be changed as the upper section does not correspond to the sky anymore.

Finally, I noticed it is relatively more difficult to detect the whole white car with respect to the whole black car. HOG features of the white car are apparently not as clear as those for the black car. I suspect that this can be even worse if the white car is contrasted with a white background. In this situation the current project pipeline is likely to fail.


---
## Resources
* Udacity project assignment and template on [GitHub](https://github.com/udacity/CarND-Vehicle-Detection)
* Project [rubric](https://review.udacity.com/#!/rubrics/513/view) Points
* [Vehicle Dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* [Non Vehicle Dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
* Additional [vehicle dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) for training
