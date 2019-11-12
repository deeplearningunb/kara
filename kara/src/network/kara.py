# Importing Libraries
from data.manager import get_images
from network.utils import (build_model, image_generator)
from keras.callbacks import TensorBoard
from skimage.color import lab2rgb, rgb2gray, rgb2lab
from skimage.io import imsave
import logging


class Kara():
    # Limit of files to be read
    FILELIMIT = 999

    def __init__(self):
        pass

    def assemble(self, batch_size: int):
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
        # Build the neural network
        logging.info('[INFO] Building neural network')
        model = build_model(
            'rmsprop',
            'mse',
            ['accuracy', 'categorical_accuracy']
        )

        # Training
        tensorboard = TensorBoard(log_dir='output/logs/first_run')
        model.fit_generator(
            image_a_b_gen(batch_size),
            callbacks=[tensorboard],
            epochs=1,
            steps_per_epoch=10
        )

        # Save model
        model_json = model.to_json()
        with open("output/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        # Test images
        X_test = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
        X_test = X_test.reshape(X_test.shape + (1,))
        Y_test = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
        Y_test = Y_test / 128
        print(model.evaluate(X_test, Y_test, batch_size=batch_size))