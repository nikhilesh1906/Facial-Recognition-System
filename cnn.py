# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
import os
import matplotlib.pyplot as plt


from sklearn.cross_validation import train_test_split

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#    classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())  #w this converts our 3D feature maps to 1D feature vectors
classifier.add(Dense(units = 64 , activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(28 , activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/kaushik/Desktop/fyp/6-4-2018/yale_openface_libsvm/aligned-images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = test_datagen.flow_from_directory('/home/kaushik/Desktop/fyp/6-4-2018/yale_openface_libsvm/validation',
                                            target_size = (64, 64),
                                            batch_size = 16,
                                            class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/home/kaushik/Desktop/fyp/6-4-2018/yale_openface_libsvm/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = 7000,
                         epochs = 14,
                         validation_data = validation_set)
                         #validation_steps = len(validation_set))

scoreSeg = classifier.evaluate_generator(test_set,len(test_set))
print("Accuracy = ",scoreSeg[1])


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



classifier.save_weights('face_weights.h5')

y_pred = classifier.predict_generator(test_set)
print(y_pred.shape,test_set)


# for root, dirs, files in os.walk("./fish_species", topdown=False):
#     if root == "./fish_species":
#         for name in dirs:
#             X =y = os.listdir(root+"/"+name)
            
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # Creating the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
