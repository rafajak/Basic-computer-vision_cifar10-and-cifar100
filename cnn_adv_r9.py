from deepsense import neptune
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from helpers import NeptuneCallback, load_cifar10, model_summary

ctx = neptune.Context()
ctx.tags.append('cnn')
ctx.tags.append('adv')

# create neural network architecture
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.6))

#add extra layers
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10))


model.add(Activation('softmax'))



model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_summary(model)

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar10('aux/cifar10.h5')

# training
model.fit(x_train, y_train,
          epochs=100,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20)])

'''
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 64)        4160      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 128)       16512     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 256)         295168    
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 256)         65792     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 256)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 512)         1180160   
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 512)         262656    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 512)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 2, 2, 512)         2048      
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 2, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1049088   
_________________________________________________________________
dropout_5 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              525312    
_________________________________________________________________
dropout_6 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                10250     
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 3,488,586
Trainable params: 3,486,666
Non-trainable params: 1,920
_________________________________________________________________


Log-loss training: 0.406463
Log-loss validation: 0.458626
Accuracy training: 0.86962
Accuracy validation: 0.8547

'''