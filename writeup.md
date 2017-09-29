
# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images.

The code for this step is contained in the second and third code cells (functions `get_hog_features` and `extract_features`) of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car Image](/output_images/car_original.png)
![Non Car Image](/output_images/not_car_original.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Car Image](/output_images/car_original.png)
![Car HOG Channel1](/output_images/car_hog1.png)
![Car HOG Channel2](/output_images/car_hog2.png)
![Car HOG Channel3](/output_images/car_hog3.png)

---
![Non Car Image](/output_images/not_car_original.png)
![Non Car HOG Channel1](/output_images/not_car_hog1.png)
![Non Car HOG Channel2](/output_images/not_car_hog2.png)
![Non Car HOG Channel3](/output_images/not_car_hog3.png)

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters and found, that the best combination for me is using:
- color space "YUV"
- orientation = 11 (bigger number is a reason to increase cardinality of feature vector, because at the and I use only HOG features)
- number of pixels per cell = 16
- number of cells per block = 2
- using of all image color space channels

This combination gave me the best classification result approx. 98% on train/test data and good estimation results.

#### 3. A classifier training.

As a classifier I choose linear Supper Vector Machines classifier.
I trained a linear SVM using only HOG features, because other combination gave me very promissing results on training and test data, but not good results (a lot of false positives) on image/video data. So, that was overfitting, and I decided, that was because training images have lot of combination of color itself, color deepness, intensity, brightness, etc.
Therefore I used only HOG features extractions (function `extract_features` in third cell).
Moreover, I did not use `StandardScaler` as well, because color histogramm and spatial features were not used.

---

### Sliding Window Search

#### 1. Description of searching areas and scales for finding cars

Function of searching cars is called `find_cars`. It is a function from lesson with some modifications.
First of all, usage of color histogramm and spatial features was deleted.
Then I decided also to introduce some borders by X-dimension on left and right sides (right side border after some experimentations was not used) to avoid false positives.
Left side border is still in use to do not find cars in opposite direction.

I decided to search window positions starting at approx. image center by Y-dimension and try several scales from small (far away) to big (close to car) values came up with this:
- several locations close to image center with scale 1
- several locations starting at the same position and ending below by increasing scale at each step
```python
locations = []
locations.append({ 'ystart': 400, 'ystop': 465, 'scale': 1.0 })
locations.append({ 'ystart': 420, 'ystop': 480, 'scale': 1.0 })
locations.append({ 'ystart': 400, 'ystop': 500, 'scale': 1.5 })
locations.append({ 'ystart': 430, 'ystop': 530, 'scale': 1.5 })
locations.append({ 'ystart': 400, 'ystop': 530, 'scale': 2.0 })
locations.append({ 'ystart': 430, 'ystop': 560, 'scale': 2.0 })
locations.append({ 'ystart': 400, 'ystop': 560, 'scale': 3.0 })
locations.append({ 'ystart': 460, 'ystop': 660, 'scale': 3.0 })
locations.append({ 'ystart': 400, 'ystop': 600, 'scale': 3.5 })
locations.append({ 'ystart': 465, 'ystop': 660, 'scale': 3.5 })
```

![All Search Areas](/output_images/all_search_areas.jpg)

#### 2. Searching pipeline demonstration

Ultimately I searched on 5 scales using LUV 3-channel HOG features, which provided a nice result.  Here are some example images:

![Find Cars](/output_images/find_cars_illustration.jpg)

---

### Video Implementation

#### 1. Final video output

Here's a [link to my short test video result](./output_videos/test_video.mp4)

Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Description of frame processing.

I recorded the positions of positive detections in each frame of the video in a global list `hist_rects`. By adding new rectangle the maximum length (20) is checked to avoid memory consumption. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Threshold is dynamic depended from records in a global list. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six images with found rectangles and their corresponding heatmaps:

![Find Cars](/output_images/find_cars_illustration.jpg)
![Heat Maps](/output_images/heat_illustration.jpg)

### Here are results of applying threshold:

![Heat Maps](/output_images/heat_tr_illustration.jpg)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six images:

![Heat Maps](/output_images/label_illustration.jpg)

### Here the resulting bounding boxes are drawn onto all images:

![Heat Maps](/output_images/final_illustration.jpg)

### Including advanced lane finding output

Additionally I added output of lane detection from [Advanced Lane Lines project](https://github.com/VolodymyrIvanov/Udacity-Advanced-Lane-Lines)

---

### Discussion

#### 1. Briefly discuss of possible problems and/or improvements

1. Some difficulties and heuristic decisions were done during playing and choising search areas and scales in image. In some cases cars can not be detected, but in some cases lot of false positives detections occured.
2. This process can be improved by specifying also borders in X-dimension on both sides. May be it can be done dynamically based on position of lane observation from [Advanced Lane Lines project](https://github.com/VolodymyrIvanov/Udacity-Advanced-Lane-Lines).
3. Scales step can be also adjusted more granular, but it can cost decreasing of perfomance. 
