import argparse
import sys
import tensorflow as tf
import numpy as np
import json


batch_size = 32
img_height = 224
img_width = 224



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="The file to predict on")
    parser.add_argument("-m", "--model", help="The trained model to use")
    args = parser.parse_args()

    if not args.file:
        print("Error: the file argument is required. Use the -f or --file flag to specify the file to predict on.")
        sys.exit()

    if not args.model:
        print("Error: the model argument is required. Use the -m or --model flag to specify the model to use.")
        sys.exit()
        
    model = load_model(args.model)

    predict(args.file, model)

def load_model(path):
    return tf.keras.models.load_model(path)

def get_class_names(): 
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
        return class_names
    

def predict(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_names = get_class_names()

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == '__main__':
    main()