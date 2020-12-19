
## SAI GUDIVADA

## THE GOAL OF THE PROJECT IS TO TRAIN A MACHINE LEARNING MODEL TO AUTOMATICALLY DETECT THE NUMBER FROM A GIVEN INPUT IMAGE


## import the necessary libraries

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import model_selection

# read the training data - which is a csv file consisiting of pixel values of the images - each image is of the shape 28 * 28
train = pd.read_csv("train.csv")
print('shape of the training data: ',train.shape)
 
 
 # divide the train data into image-data and output label data
sample_images = train.values[:, 1:]
sample_labels = train.values[:, :1]
print(sample_labels.shape, sample_images.shape)
 
 
 # split the data into 80-20 ratio to train the model and validate on the validation data
train_images, validation_images, train_labels, validation_labels = model_selection.train_test_split(sample_images, sample_labels, train_size=0.8, random_state=0)

print('shape of the train_images, validation_images, train_labels, validation_labels: ', train_images.shape, train_labels.shape, validation_images.shape, validation_labels.shape)

# sample input image and its histogram of the pixel values
i = 5
show_image = np.asmatrix(train_images[i])
show_image = show_image.reshape((28, 28))
plt.imshow(show_image)
plt.pause(5)
plt.hist(show_image)
plt.pause(5)


#making the image binary
train_images[train_images>0] = 1
validation_images[validation_images>0] = 1

#train the SVM model
clf_bin = svm.SVC()
clf_bin.fit(train_images, train_labels.ravel())

#check how good the model is performing on the validation data
print(clf_bin.score(validation_images, validation_labels))

#read the test data which has unknown labels and let us see how is the model predicting the number by looking at the input image 
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
print(test_data.shape)
print(len(test_data.iloc[:,0]))
print(test_data.iloc[0,:].shape)
for i in range(len(test_data.iloc[:,0])):
    test_image = np.asarray(test_data.iloc[i,:])
    test_image = test_image.reshape((28, 28))
    plt.imshow(test_image)
    plt.pause(5)
    print(clf_bin.predict(test_data[i]))
    print('\n\n')





