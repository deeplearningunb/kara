from keras.applications.inception_resnet_v2 import (InceptionResNetV2,
                                                    preprocess_input)
from keras.layers.core import (RepeatVector, Reshape)
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Conv2D, UpSampling2D, Input, concatenate)
from keras.models import (Model)
from skimage.color import (gray2rgb, rgb2lab, rgb2gray)
from skimage.transform import (resize)
import numpy as np
import tensorflow as tf


def build_model(optimizer: str, loss: str, metrics: list):
    embed_input = Input(shape=(1000,))

    # Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3, 3), activation='relu',
                            padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3, 3), activation='relu',
                            padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu',
                            padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu',
                            padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu',
                            padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu',
                            padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu',
                            padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu',
                            padding='same')(encoder_output)

    # Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu',
                           padding='same')(fusion_output)

    # Decoder
    decoder_output = Conv2D(128, (3, 3), activation='relu',
                            padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu',
                            padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu',
                            padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation='relu',
                            padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh',
                            padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(
        inputs=[encoder_input, embed_input],
        outputs=decoder_output
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def create_inception_embedding(grayscaled_rgb: list, inception):
    grayscaled_rgb_resized = []
    for obj in grayscaled_rgb:
        obj = resize(obj, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(obj)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


def load_weights(weights: str):
    inception = InceptionResNetV2(
        weights=weights,
        include_top=True
    )
    inception.graph = tf.compat.v1.get_default_graph()
    return inception


def image_generator(batch_size: int, base_images: list, inception):
    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True
    )
    # Generate images
    for batch in datagen.flow(base_images, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb, inception)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        X_batch = X_batch.reshape(X_batch.shape + (1,))
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield ([X_batch, embed], Y_batch)
