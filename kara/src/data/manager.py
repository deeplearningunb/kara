from keras.preprocessing.image import (img_to_array, load_img)
import os
import numpy as np
import logging


def get_images(path: str, limit: int):
    images = []
    c = 1
    for file in os.listdir(path):
        # Reducing number of images until optimization
        if c > limit:
            break
        logging.debug(f'[DEBUG] Getting image[{c}] -- {file}')
        images.append(
            img_to_array(
                load_img(path + file)
            )
        )
        c += 1

    images = np.array(images, dtype=float)
    return images

def get_custom_images(images_list):
    images = []
    for c, file in enumerate(images_list):
        logging.debug(f'[DEBUG] Getting image[{c}] -- {file}')
        images.append(
            img_to_array(
                load_img(file)
            )
        )

    images = np.array(images, dtype=float)
    return images