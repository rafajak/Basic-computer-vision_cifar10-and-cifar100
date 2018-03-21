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

model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                 input_shape=(32, 32, 3)))
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
model.add(Dense(10))
model.add(Activation('softmax'))

#Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
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
