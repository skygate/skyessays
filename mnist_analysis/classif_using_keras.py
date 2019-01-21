from random import randint

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from numpy.random import seed

import keras

from utils import keras_display_prediction

seed(3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

figure = plt.figure()
for x in range(1, 101):
    plt.subplot(10, 10, x)
    plt.axis('off')
    plt.imshow(x_train[randint(0, 55000)], cmap=plt.get_cmap('gray_r'))
plt.savefig("digits.png", bbox_inches='tight')

vector = 784

x_train = x_train.reshape(60000, vector)
x_test = x_test.reshape(10000, vector)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Dense(784, activation='relu', input_shape=(vector,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

batch_size = 500
epochs = 20

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1)

test_loss, test_acc = model.evaluate(x_test, y_test)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
plt.savefig("accuracy.png", bbox_inches='tight')

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

figure = plt.figure()
for x in range(1, 10):
    plt.subplot(3, 3, x)
    plt.axis('off')
    keras_display_prediction(randint(0, 55000), x_train, y_train, model)
plt.show()
plt.savefig("prediction.png", bbox_inches='tight')
