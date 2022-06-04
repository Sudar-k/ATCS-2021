#import libraries

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch

# these lines ensure that I can open a URL without having a verified certification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Sources Used
# For sample discriminator and generator: https://www.tensorflow.org/tutorials/generative/dcgan

# For face detection with OpenCV: https://realpython.com/face-recognition-with-python/
                                # https://github.com/shantnu/FaceDetect

# For using webcam: https://realpython.com/face-detection-in-python-using-a-webcam/

# For face detection with MTCNN: https://github.com/ResByte/mtcnn-face-detect
# https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

# For cartoonification of faces: https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e
# https://github.com/tazkianida/cartoon-effect

# For anime-ification of faces: https://github.com/TachibanaYoshino/AnimeGANv2 (didn't use)
# https://github.com/bryandlee/animegan2-pytorch (used this)



#Creating sample discriminator for GAN

# discriminator = tf.keras.Sequential()
# discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[28, 28, 1]))
# discriminator.add(layers.LeakyReLU())
# discriminator.add(layers.Dropout(0.3))
#
# discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
# discriminator.add(layers.LeakyReLU())
# discriminator.add(layers.Dropout(0.3))
#
# discriminator.add(layers.Flatten())
# discriminator.add(layers.Dense(1))
#
# #Creating sample generator for GAN
#
# generator = tf.keras.Sequential()
# generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
# generator.add(layers.BatchNormalization())
# generator.add(layers.LeakyReLU())
#
# generator.add(layers.Reshape((7, 7, 256)))
# assert generator.output_shape == (None, 7, 7, 256)
#
# generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
# assert generator.output_shape == (None, 7, 7, 128)
# generator.add(layers.BatchNormalization())
# generator.add(layers.LeakyReLU())
#
# generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
# assert generator.output_shape == (None, 14, 14, 64)
# generator.add(layers.BatchNormalization())
# generator.add(layers.LeakyReLU())
#
# generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
# assert generator.output_shape == (None, 28, 28, 1)


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

#use webcam to take picture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Video", frame)

    #click 's' key to take picture
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # take picture and save as webcam.jpg
        cv2.imwrite("webcam.jpg", frame)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

#Face Recognition with MTCNN

ax = plt.gca()

def draw_image_with_boxes(filename, result_list):
    images = []
    coords = []
    # load the image
    counter = 0

    data = Image.open(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes

    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)

        # crops the faces and saves each of them as an image
        a = data.crop((x, y, x + width, y + height))
        img_name = "face_{}.jpg".format(counter)
        a.save(img_name)
        counter += 1
        images.append(img_name)
        coords.append((x, y, width, height))

    # show the plot
    plt.show()
    #returns names of images of faces as well as coordinates, width, height of original images
    return images, coords


# Cartoonifying Image
def color_quantization(img, k):
# Transform the image
  data = np.float32(img).reshape((-1, 3))

# Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # blur to eliminate unwanted aspects of the image
  gray_blur = cv2.medianBlur(gray, blur_value)

  # detect the edges of the image
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges



def image_to_cartoon(file):
    img = cv2.imread(file)

    # sets the line size of edges and the blur value
    line_size = 7
    blur_value = 7

    edges = edge_mask(img, line_size, blur_value)

    # k value
    total_color = 9

    img = color_quantization(img, total_color)

    # add bilateral filter to reduce noise in the image
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=150,sigmaSpace=200)

    # combine edge mask with color processed image
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

# sets the webcam as the image to convert, change this to "ppl.jpeg" to test multiple people
filename = 'webcam.jpg'
pixels = plt.imread(filename)

detector = MTCNN()
faces = detector.detect_faces(pixels)

imgs, crds = draw_image_with_boxes(filename, faces)

mode = "anime"

if mode == "cartoon":
    cartoons = []
    counter = 0
    # transforms each image of a face into a cartooned version
    for i in imgs:
        cartoon = image_to_cartoon(i)
        img_name = "cartoon_{}.jpg".format(counter)
        cv2.imwrite(img_name, cartoon)
        counter += 1
        cartoons.append(img_name)

    # pastes the now cartoonified faces back onto the original image
    img1 = Image.open(filename)
    for i in range(len(cartoons)):
        c_img = Image.open(cartoons[i])
        img1.paste(c_img, (crds[i][0], crds[i][1]))

    #deletes cartoon-ified images
    for c in cartoons:
        os.remove(c)

elif mode == "anime":
    #loads pretrained model from github with anime style 'Paprika'
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")

    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

    animes = []
    counter = 0
    for i in imgs:
        input = Image.open(i).convert('RGB')

        #uses face2paint function to apply model on images of faces
        anime = face2paint(model, input)
        img_name = "anime_{}.jpg".format(counter)

        #resizes anime-ified image so it fits in original image
        resized = anime.resize((crds[counter][2], crds[counter][3]))
        resized.save(img_name)

        counter += 1
        animes.append(img_name)

    #pastes anime-ified images back into original image
    img1 = Image.open(filename)
    for i in range(len(animes)):
        c_img = Image.open(animes[i])
        img1.paste(c_img, (crds[i][0], crds[i][1]))

    #deletes anime-ified images
    for a in animes:
        os.remove(a)

# show final product
img1.show()

# deletes individual images of face and original image
for f in imgs:
    os.remove(f)
os.remove("webcam.jpg")