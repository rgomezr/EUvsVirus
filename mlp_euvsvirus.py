import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import image
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import optimizers
from sklearn.model_selection import train_test_split # used for splitting the data set into train and test set

# main

# getting the path of the csv file
filename = '/Users/rgomezr/Documents/OneDrive/DOCUMENTS/workspace/EUvsVirus/datasets/dataset_1.csv'

# getting all students data in the csv in a panda data frame matrix
students_data = pd.read_csv(filename, float_precision= 'round_trip')

# retrieving columns of interest for X input data
cols = [col for col in students_data.columns if col not in ['output_label']]

# retrieving columns of interest for X input data
data = students_data[cols]
print(data)

# generating the column vector containing all output labels
labels = students_data['output_label']

# splitting the data into training and test datasets
train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size = 0.20, random_state = 10)

print(train_X.shape, train_Y.shape)

# Multilayer-Perceptron Model structure
model = Sequential()
model.add(Dense(16, input_dim=12))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])

"""
Creating validation data based on test data to validate the model
agains each iteration of the model
"""
validation_data = (test_X, test_Y)

# training the model
h = model.fit(train_X, train_Y, epochs = 120, batch_size = 150, validation_data = validation_data)


# Score of the accuracy and the loss based on test/unseen data
score = model.evaluate(test_X, test_Y, batch_size=128)

print("final testing score: ", model.metrics_names, score)


# plotting graphs to evaluate the model

accuracies = h.history['accuracy']
acc = plt.figure(1)
plt.plot(np.squeeze(accuracies))
plt.xlabel('Epocs')
plt.ylabel('Accuracy')


losses = h.history['loss']
loss = plt.figure(2)
plt.plot(np.squeeze(losses))
plt.xlabel('Epocs')
plt.ylabel('Loss')

val_accuracies = h.history['val_accuracy']
val_acc = plt.figure(1)
plt.plot(np.squeeze(val_accuracies), color = 'red')



val_losses = h.history['val_loss']
val_loss = plt.figure(2)
plt.plot(np.squeeze(val_losses), color='red')


plt.show()







