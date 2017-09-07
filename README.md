# Geometric-shapes-classification-MATLAB-

It is the simplest classification model for the geometric shapes(Triangle, Rectangle, and Circles). Image of any geometric shape(Triangle, Rectangle, and Circles) is taken as input, and it predicts the numeric label corresponding to that shape,
<<<<<<< HEAD
{(1->Triangle),(2->Circle),(3->Rectangle)}.
=======
     {(1->Triangle), (2->Circle), (3->Rectangle)}.
>>>>>>> c541054c08ec3a042c27909e45be0e6cda0b411b
It involves Feature extraction and  Classification model (neural networks): 
(1) Feature extraction is used to extract the features from the provided image (number of Extracted features may varies from image to image) and reduce the dimensionality of feature set, in order to get a fixed number of features for each image.
(2) Classification model takes the transformed features of the image dataset( 5 principal components) as inputs and outputs a label accordingly.

## Dataset
Shapes folder contain 60 images of different shapes(triangle,circle,rectangle), 20 of each class.

## File System
 - PCA_shapes.m : Script to Extract features from images
 - neural_shapes.m : Script for Classification model
 - shapes.mat : Saved workspace from earlier run
 - pca_shapes.csv : Extracted features used as input for classification model
 - shapes(folder) : Contains images numbered from 1 to 60
 - shaples_label : File storing the targets(labels) for the images dataset 

## Author
Sourabh Garg sourabh.8.june.1996@gmail.com @codeSG

