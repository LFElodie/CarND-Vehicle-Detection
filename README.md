## README

**Vehicle Detection Project**

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
- Normalize features and randomize a selection for training and testing.
- Implement a sliding-window technique and use trained classifier to search for vehicles in images.
- Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Extraction of HOG features, spatially binned color and histograms of color from the training images.

The code for this step is contained in the ` visualization.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of the `vehicle` and `non-vehicle` classes:

![](http://p37mg8cnp.bkt.clouddn.com/github/md/cars_notcars.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![](http://p37mg8cnp.bkt.clouddn.com/github/md/hog_show.png)

Here is an example of histogram of color using the `RGB` color space and `hist_bins=32`

![](http://p37mg8cnp.bkt.clouddn.com/github/md/histogram.png)

Here is an example of spatial binning of color using the `RGB` color space and `spatial_size=32`

![](http://p37mg8cnp.bkt.clouddn.com/github/md/spatial.png)

#### 2. Parameters for HOG features, histogram and spatial features.

The code for this step is contained in the `Data preparation` and`Training classifier`  of the ` vehicle_detection.ipynb`.

I tried various combinations of parameters and trained a linear SVM.

| Trial | Colorspace | Orient | Pixels/cell | Cells/Block | length | Accuracy |
| ----- | ---------- | :----: | :---------: | :---------: | :----: | :------: |
| #1    | RGB        |   9    |      8      |      2      |  8460  |  98.68%  |
| #2    | HLS        |   9    |      8      |      2      |  8460  |  99.18%  |
| #3    | HSV        |   9    |      8      |      2      |  8460  |  99.35%  |
| #4    | YUV        |   9    |      8      |      2      |  8460  |  99.30%  |
| #5    | YCrCb      |   9    |      8      |      2      |  8460  |  99.32%  |
| #6    | YCrCb      |   8    |      8      |      2      |  7872  |  99.21%  |
| #7    | YCrCb      |   10   |      8      |      2      |  9048  |  99.21%  |
| #8    | YCrCb      |   9    |     16      |      2      |  4140  |  99.07%  |
| #9    | YCrCb      |   9    |      4      |      2      | 27468  |  99.10%  |
| #10   | YCrCb      |   9    |      8      |      4      | 13968  |  99.24%  |

I finally chose

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 4 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

Using HSV, YUV,  YCrCb color space all can get high accuracy but YCrCb color space can get fewer false positive when searching on the image. Using other parameters may be harmful to the results or may not improve the results and only increased the feature vector. That is why these parameters were chosen.

#### 3. Training a classifier.

I trained a linear SVM with the default classifier parameters and using HOG features with parameters describe above. I got about 0.993 accuracy on the test set.

### Sliding Window Search

#### 1. What scales to search and how much to overlap windows.

I decided to search random window positions at random scales all over the image and came up with this.

![](http://p37mg8cnp.bkt.clouddn.com/github/md/slidingwindow2.png.png)

(ok just kidding I didn't actually ;):

Define pixels per cell as 8 x 8, then a scale of 1 would retain a window that's 8 x 8 cells (8 cells to cover 64 pixels in either direction). An overlap of each window can be defined in terms of the cell distance, using  `cells_per_step`. This means that a `cells_per_step = 2` would result in a search window overlap of 75% (2 is 25% of 8, so we move 25% each time, leaving 75% overlap with the previous window). Any value of scale that is larger or smaller than one will scale the base image accordingly, resulting in corresponding change in the number of cells per window. I search at  `scale = 1.0, 1.3, 1.7, 2.0 ` scaled search windows and came up with this.

![](http://p37mg8cnp.bkt.clouddn.com/github/md/searchstrategy.png)

#### 2. Performance on test images

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![](http://p37mg8cnp.bkt.clouddn.com/github/md/result1.png)

![](http://p37mg8cnp.bkt.clouddn.com/github/md/result2.png)

![](http://p37mg8cnp.bkt.clouddn.com/github/md/result3.png)

### Video Implementation

#### 1. Here's a [link to my video result](./project_video.mp4)

#### 2. Some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![](http://p37mg8cnp.bkt.clouddn.com/seriesheatmap.png)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

Before filter.

![](http://p37mg8cnp.bkt.clouddn.com/seriesheatgray.png)

After filter.

![](http://p37mg8cnp.bkt.clouddn.com/filtedheatmap.png)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![](http://p37mg8cnp.bkt.clouddn.com/github/md/filtered_result.png)

I apply a filter on the heatmap and the tracker.Add a deque to store history of the heatmap, and set the threshold to `2*len(history)`, only detected twice consecutively is considered to be detected. That can reduce false positives. I also apply a smoother on the tracker to smooth the bounding box.

---

### Discussion

#### 1. Problems and outlook

Problems

1. The problems that I faced while implementing this project were mainly concerned with balancing the accuracy with execution speed.  Extracting more features or searching more windows may help improve accuracy but greatly increase execution time.
2. Decide which car the detected bounding box belongs to when the cars overlap is also a problem.
3. The pipeline is probably most likely to fail in cases where vehicles don't resemble those in the training dataset, and lighting environment might also have a great impact. 

Outlook

1. Use multi-threaded parallel to increase execution speed.
2. Write a more reliable tracker.
3. Use more dataset to train a better classifier.
4. Try CNN like YOLO or Faster R-CNN to get a more robust performance.