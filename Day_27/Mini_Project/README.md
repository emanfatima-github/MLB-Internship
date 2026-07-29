# Image Segmentation Project

## What is Image Segmentation?

Image segmentation is the process of separating the main object or document from the background so that different parts of the image can be identified clearly.

## Difference between Binary, Adaptive, and Otsu Thresholding

* Binary Thresholding: Uses one fixed threshold value for the entire image. It works well when the lighting is even.
* Adaptive Thresholding: Uses different threshold values for different areas of the image. It performs better on images with uneven lighting or shadows.
* Otsu Thresholding: Automatically finds the best threshold value without manually selecting it. 
It is useful when the foreground and background have clear intensity differences.

## Which Method Worked Best and Why?

For our dataset, Otsu Thresholding gave the best overall results because it automatically selected an appropriate threshold and 
produced cleaner segmentation for most document and object images. Adaptive Thresholding performed better on images with shadows or uneven lighting.

## Challenges Faced During Implementation

Choosing a single threshold value for all images was difficult, 
so different thresholding methods were compared to achieve better results.
