# Content Based Image Classification
CBIC Using KNN Classifier

The aim of this project is image classification with three steps: training, validation and testing. The main purpose is to reach the optimal case (highest accuracy) with respect to changing number of bins and grid levels. The system is trained by the training feature, accuracy of the system is calculated by validation feature and system can be tested by testing feature.

## CBIC
The CBIC system pipeline starts with feature extraction of the query image and all other images in the database. After obtaining all the features, a similarity test is applied between the features of each image in the database and the query image. Finally, based on the result of classification, the most similar images are identified and assigned to appropriate class label.

## Histograms
A histogram is a vector that counts how many instances of a given property exist in the image. First step of creating a histogram is to define ranges to determine bins, then each instance is assigned into one of these bins. 
### Grayscale intensity histogram: 
This is simply obtained by quantizing the pixels into histogram bins based on their intensity level and then computing the frequency of each intensity bin in the image.
### Color histogram: 
The color channel histogram can be obtained by quantizing pixels at each color channel separately and then assigning pixels into combination of bins of these three histograms

## Grid Based Histogram Feature Extraction
Every level corresponds to grid division of an image. For example, level 1 which results in a single histogram; or we can divide the image into a grid and extract the histogram for each cell of the grid individually, and then concatenate the resulting histograms. For this project, level 1 corresponds to constructing a 1x1 grid level, 2 corresponds to constructing a 2x2 grid and level 3 corresponds to constructing a 4x4 grid. 

## Classification
Classification is done using KKN classifier and Euclidean distance measure. Multiple K values have been tested.

## Programming
This project requires implementing the aforementioned CBIC system using histograms of grayscale and color intensity using different spatial levels, using euclidean distance and different values of K.
### Training:
System is trained by the images which are in Train folder.
### Validating:
System validation is for calculating the accuracy of the program and images in the Validate folder are used.
### Testing:
The images in Test folder are for testing.

## Database
Database contains 3 classes (cloudy, shine and sunrise) and consists of 30 images. A query is simply the name of an image whose content will be used for classification. For these queries, the ground truth results (class labels) are the names of the images.

## Files
GrayScaleCode.py : Grayscale Training and Validation <br />
RgbHistogramCode.py : RGB Training and Validation <br />
TestImagesTestCode.py : Testing the best case in all validations

