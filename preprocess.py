import tensorflow as tf
from PIL import Image
import numpy as np
import os

def preprocess_image(image_path):
    label = get_labels(image_path)

    # Read the image from file
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string)

    # Resize and crop the image to a fixed size
    resized_image = tf.image.resize(image_decoded, [224, 224])

    # Normalize the pixel values of the image
    normalized_image = tf.image.per_image_standardization(resized_image)

    print(normalized_image.shape)
    
    return normalized_image, label




def get_labels(file_path):
    return tf.strings.split(file_path , '/')[-2]





def re_scale(img, labels):
    return img/255 , labels

# def preprocess_image(file_path):
#     label = get_labels(file_path)

#     img = tf.io.read_file(file_path)
#     img = tf.image.decode_jpeg(img)
#     img = tf.image.resize(img, [224,224])
    
#     return img, label


def main():
    path = 'data2/acne_affecting_the_back/141__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpg'
    image  = preprocess_image(path)
    # print("image1", image)
    

    

if __name__ == '__main__':
    main()