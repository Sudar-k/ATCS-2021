#import libraries

import tensorflow as tf
import numpy as np
import PIL
import imageio
from tensorflow.python import keras
from keras import layers
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle

#https://www.tensorflow.org/tutorials/generative/dcgan
#Creating sample discriminator for GAN

discriminator = tf.keras.Sequential()
discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1))

#Creating sample generator for GAN

generator = tf.keras.Sequential()
generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Reshape((7, 7, 256)))
assert generator.output_shape == (None, 7, 7, 256)

generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
assert generator.output_shape == (None, 7, 7, 128)
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
assert generator.output_shape == (None, 14, 14, 64)
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
assert generator.output_shape == (None, 28, 28, 1)


# ''' Detect faces with OpenCV'''
#
# # imagePath = "multiple.png"
#
# cascPath = "haarcascade_frontalface_default.xml"
#
# # Create the haar cascade
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# # Read the image
# # image = cv2.imread(imagePath)'
#
# #Use Webcam to analyze frame-by-frame
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the image
#     faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=4,
#     minSize=(30, 30),
#
#     )
#
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()

#Face Recognition with MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()


filename = 'me and eli.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)

''' Using Webcam with MTCNN'''
video_capture = cv2.VideoCapture(0)
detector = MTCNN()
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

