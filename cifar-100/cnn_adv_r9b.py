from deepsense import neptune
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from helpers import NeptuneCallback, load_cifar100, model_summary
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks

ctx = neptune.Context()
ctx.tags.append('cnn')

data_augmentation = True
num_classes = 100

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5)) # added this

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

optimizer = keras.optimizers.Adam(decay=0.0001) # from 0.00006

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model_summary(model)

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar100('aux/cifar100.h5')

callbacks = [NeptuneCallback(x_test, y_test, images_per_epoch=10)]

# training
model.fit(x_train, y_train,
          epochs=100,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=callbacks)
