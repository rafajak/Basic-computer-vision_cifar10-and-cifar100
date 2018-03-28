from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization
from helpers import NeptuneCallback, load_cifar10, model_summary
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('cnn')
ctx.tags.append('adv')
ctx.tags.append('.')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='valid',
                 input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(128, (1, 1), activation='relu', padding='valid'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('softmax'))

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_summary(model)

# loading data
(x_train, y_train), (x_test, y_test) = load_cifar10('aux/cifar100.h5')

# training
model.fit(x_train, y_train,
          epochs=100,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20)])
