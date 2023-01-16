import argparse
import sys
import tensorflow as tf
import numpy as np
import json
import os

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
        
        
    # Make a prediction with the chosen file and the chosen model
    score_arr = predict(args.file, args.model)
    
    # Get the scores of the 5 best
    scores = get_scores(score_arr, 5)



    print()
    print("======================================")

    for score in scores:
        print("{}: {:.2f}%".format(score["name"].replace("_", " ").capitalize(), score['score']*100))

    print("======================================")
    print()


def get_scores(score_arr, amount_result: int = 3):
    

    # Convert to numpy array ins tead of tensor
    score_arr = score_arr.numpy()
    class_names = get_class_names()
    
    arr = []
        
    for index, score in enumerate(score_arr):
        obj = {}
        obj["name"] = class_names[index]
        obj["score"] = score
        
        arr.append(obj)
    
    
    score_arr = sorted(arr, reverse=True, key=lambda d: d['score']) 

    return score_arr[:amount_result]



def load_model(path):
    return tf.keras.models.load_model(path)


def get_class_names(): 
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
        return class_names
    

def predict(image_path: str, model_path: str):
        
    model = load_model(model_path)
    
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    
    return scores


if __name__ == '__main__':
    main()