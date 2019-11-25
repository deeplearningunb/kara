# Importing Libraries
from data.manager import get_images
from network.utils import (build_model, create_inception_embedding,
                           load_weights, image_generator)
from keras.callbacks import TensorBoard
from skimage.color import lab2rgb, rgb2gray, rgb2lab, gray2rgb
from skimage.io import imsave
import logging
import tensorflow as tf
import numpy as np
from keras.models import model_from_json


class Kara():
    # Limit of files to be read
    # Define the limit via environment variables
    FILELIMIT = 999

    def __init__(self):
        self.inception = None
        self.model = None

    def predict_test_images(self, number_of_images: int):
        X_test = get_images('dataset/images/test/', number_of_images)
        X_test = 1.0/255*X_test
        X_test = gray2rgb(rgb2gray(X_test))

        X_test_embed = create_inception_embedding(X_test, self.inception)

        X_test = rgb2lab(X_test)[:, :, :, 0]
        X_test = X_test.reshape(X_test.shape+(1,))

        output = self.model.predict([X_test, X_test_embed])
        output = output*128

        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:, :, 0] = X_test[i][:, :, 0]
            cur[:, :, 1:] = output[i]
            imsave("output/images/img_" + str(i) + ".png", lab2rgb(cur))

    def assemble(self, batch_size: int):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # Get images
        logging.info('[INFO] Getting training images...')
        # Needs to be optimized
        X = get_images('dataset/images/train/', self.FILELIMIT)
        logging.info('[INFO] All images were imported without error')

        # Split test and train data
        logging.info('[INFO] Spliting test and train data')
        split = int(len(X) * 0.95)
        X_train = X[:split]
        X_train = (1.0 / 255 * X_train)

        # Load weights
        inception = load_weights('imagenet')
        # Build the neural network
        logging.info('[INFO] Building neural network')
        model = build_model(
            'adam',
            'mse',
            ['accuracy', 'categorical_accuracy']
        )

        # Training
        tensorboard = TensorBoard(log_dir='output/logs/first_run')
        model.fit_generator(
            image_generator(batch_size, X_train, inception),
            callbacks=[tensorboard],
            epochs=1,
            steps_per_epoch=1
        )

        # Save model
        model_json = model.to_json()
        with open("output/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")

        self.model = model

        # # Test images
        # X_test = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
        # X_test = X_test.reshape(X_test.shape + (1,))
        # Y_test = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
        # Y_test = Y_test / 128
        # print(model.evaluate(X_test, Y_test, batch_size=batch_size))

    def assemble_from_file(self):
        with open('output/model.json', 'r') as f:
            model_json = f.read()

        model = model_from_json(model_json)
        model.load_weights('model.h5')

        self.model = model

        # Load weights
        inception = load_weights('imagenet')

        self.inception = inception
