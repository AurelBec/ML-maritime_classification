# Convolutional Neural Network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(64, 3, 3, input_shape = (100, 30, 3), activation = 'relu')) # Convolution layer
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Pooling layer
classifier.add(Conv2D(64, 3, 3, activation = 'relu')) # Adding a second convolutional layer
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Adding a second convolutional layer
classifier.add(Flatten()) # Flattening layer
classifier.add(Dense(units = 128, activation = 'relu')) # Full connection
classifier.add(Dense(units = 24, activation = 'sigmoid')) # Full connection

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

training_set = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True).flow_from_directory('dataset/training_set', target_size = (100, 30))
test_set = ImageDataGenerator(rescale = 1./255).flow_from_directory('dataset/test_set', target_size = (100, 30))
classifier.fit_generator(training_set, steps_per_epoch = 4774, epochs = 1, validation_data = test_set, validation_steps = 1682)


# 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

tested = 'Mototopo'
result = classifier.predict(np.expand_dims(image.img_to_array(image.load_img('dataset/single_prediction/'+tested+'.jpg', target_size = (100, 30))), axis = 0))
id = test_set.class_indices
print ('\nPrediction: {} (expected {})'.format(list(id.keys())[list(id.values()).index(np.argmax(result[0]))], tested))
