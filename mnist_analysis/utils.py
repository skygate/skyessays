import matplotlib.pyplot as plt


def keras_display_prediction(num, x_train, y_train, model):
    x_train_ex = x_train[num, :].reshape(1, 784)
    y_train_ex = y_train[num, :]
    label = y_train_ex.argmax()
    prediction = int(model.predict_classes(x_train_ex))
    plt.title(f'Predicted: {prediction} Label: {label}')
    plt.imshow(x_train_ex.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))


def tf_display_prediction(num, mnist, sess, y, X):
    x_train = mnist.train.images[num, :].reshape(1, 784)
    y_train = mnist.train.labels[num, :]
    label = y_train.argmax()
    prediction = sess.run(y, feed_dict={X: x_train}).argmax()
    plt.title(f'Prediction: {prediction} Label: {label}')
    plt.imshow(x_train.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
    plt.show()
