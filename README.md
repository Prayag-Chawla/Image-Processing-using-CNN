
# Image Processing using CNN

In this project,we have reduced the image matrix to greyscale, then shortended it using various models, performed various functions using various pythona nd deep learning libraries and functionalites of convolutional neural networks, and then finally tokenised every image into a variable and then exported it via a csv file.
In this repo, I have given two examples of two datasets, both are classic ones and easily available. I will provide the link for one of those.


## CNN

Convolutional Neural Network is a Deep Learning algorithm specially designed for working with Images and videos. It takes images as inputs, extracts and learns the features of the image, and classifies them based on the learned features.
## Description

- Usage of convolutional neural networksa for imge processing.

- various graphical approach to visualise the images.
- Usage of various greyscale images and other important fucntions.

## Usage and Installation

```
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
model = Sequential()





Sequential model
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the classifier
#defining sequential i.e sequense of layers


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,activation = 'relu'))
#units = 6 as no. of column in X_train = 11 and y_train =1 --> 11+1/2

#Adding the second hidden lyer
classifier.add(Dense(units = 6, activation='relu'))

#adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid))



```


## Acknowledgements

  [https://www.kaggle.com/code/wonjinkim1010/mnist-0-991-cnn-modelc](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)



## Appendix

A very crucial project in the realm of data science and image processing. Multiple high level concepts were used.


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

We got an accuracy of 97.8 % in this model. We ran various epochs and used efficiet data cleansing techniques to get to this.

## Used By

The project is used by a lot of social media companies to analyse their market.


## OUTPUT
![image](https://user-images.githubusercontent.com/92213377/215312204-137b9f19-feed-4fe6-95bc-3f45c97f3d38.png)
![image](https://user-images.githubusercontent.com/92213377/215312221-179526d0-5c6a-4fdc-9a89-196534401b13.png)
![image](https://user-images.githubusercontent.com/92213377/215312322-62753467-0eac-4fba-a0bd-19469af14139.png)
![image](https://user-images.githubusercontent.com/92213377/215312349-78bb4f91-5a0d-41b7-9cdd-e9902cd071a5.png)







