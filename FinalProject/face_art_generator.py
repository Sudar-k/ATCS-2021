#import libraries

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

#https://www.tensorflow.org/tutorials/generative/dcgan
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

#Face Recognition with MTCNN

ax = plt.gca()

def draw_image_with_boxes(filename, result_list):
    images = []
    coords = []
    # load the image
    counter = 0
    # data = pyplot.imread(filename)
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

        a = data.crop((x, y, x + width, y + height))
        img_name = "face_{}.jpg".format(counter)
        a.save(img_name)
        counter += 1
        images.append(img_name)
        coords.append((x, y))

    # show the plot
    plt.show()
    return images, coords


#
#
# filename = 'me and eli.jpg'
# # load image from file
# pixels = pyplot.imread(filename)
# # create the detector, using default weights
# detector = MTCNN()
# # detect faces in the image
# faces = detector.detect_faces(pixels)
# # display faces on the original image
# draw_image_with_boxes(filename, faces)


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("pic.jpg", frame)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

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
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges



def image_to_cartoon(file):
    img = cv2.imread(file)
    line_size = 7
    blur_value = 7

    edges = edge_mask(img, line_size, blur_value)


    total_color = 13

    img = color_quantization(img, total_color)


    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)


    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon


filename = 'ppl.jpeg'
pixels = plt.imread(filename)

detector = MTCNN()
faces = detector.detect_faces(pixels)

imgs, crds = draw_image_with_boxes(filename, faces)

cartoons = []
counter = 0
for i in imgs:
    cartoon = image_to_cartoon(i)
    img_name = "cartoon_{}.jpg".format(counter)
    cv2.imwrite(img_name, cartoon)
    counter += 1
    cartoons.append(img_name)


# cartoon = image_to_cartoon('face_0.jpg')
# cv2.imwrite("cartoon.jpg", cartoon)
img1 = Image.open(filename)
for i in range(len(cartoons)):
    c_img = Image.open(cartoons[i])
    img1.paste(c_img, crds[i])



# cartoon_img = Image.open('cartoon.jpg')

# img1.paste(cartoon_img, crds[0])
# cv2.imshow()
img1.show()

print("hello world")
# deletes picture after it's been taken

for c in cartoons:
    os.remove(c)

for f in imgs:
    os.remove(f)
os.remove("pic.jpg")
