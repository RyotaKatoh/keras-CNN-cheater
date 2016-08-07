from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import sys
import glob
from PIL import Image
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '../deep-learning-models'))
import vgg16

NUM_BATCH = 5


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=256 * 256 * 3))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('relu'))
    model.add(Reshape((256, 256, 3)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 5, 5, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def generator_with_classifier(generator, classifier):
    model = Sequential()
    model.add(generator)
    classifier.trainable = False
    model.add(classifier)

    return model


def generate(filepath, output_dir, BATCH_SIZE):
    generator = generator_model()
    adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='categorical_crossentropy', optimizer=adam)
    generator.load_weights(filepath)

    noise = np.zeros((BATCH_SIZE, 100))
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)

    print('Generating images..')
    generated_images = [np.rollaxis(img, 0, 3)
                        for img in generator.predict(noise)]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for index, img in enumerate(generated_images):
        img = img * 127.5
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, "{0}.jpg".format(index)))


def train(target_class, BATCH_SIZE):

    generator = generator_model()
    classifier = vgg16.VGG16(include_top=True, weights='imagenet')
    classifier_on_generator = generator_with_classifier(
        generator=generator, classifier=classifier)

    adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='categorical_crossentropy', optimizer=adam)
    classifier_on_generator.compile(
        loss='categorical_crossentropy', optimizer=adam)

    for epoch in range(NUM_BATCH):
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)

        y = np_utils.to_categorical([target_class] * BATCH_SIZE, 1000)

        g_loss = classifier_on_generator.train_on_batch(noise, y)
        print("Generator loss: ", g_loss)

        if epoch == NUM_BATCH - 1:
            print('saving weights...')
            filename = "{0}_generator.h5".format(target_class)
            generator.save_weights(filename, True)

            generate(filename, target_class, 10)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.mode not in ("train", "generate"):
        raise ValueError("mode must be 'train' or 'generate'")
    train(args.target_class, args.batch_size)
