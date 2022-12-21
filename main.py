# Import libraries for working with datasets
import os
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from preprocess import preprocess_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # Create a dataframe for the labeled images
    df = pd.DataFrame(columns=['file', 'label', 'image'])

    data_path = 'data2/'

    # Loop through the directories containing the images and add them to the dataframe
    for disease in os.listdir(data_path):
        for filename in os.listdir(os.path.join(data_path, disease)):
            df_dictionary = pd.DataFrame([{'file': os.path.join(data_path, disease, filename), 'label': disease}])
            df = pd.concat([df, df_dictionary], ignore_index=True)

    # Loop through the dataframe and preprocess the images
    for index, row in df.iterrows():
        df.loc[index, 'image'] = preprocess_image(row['file'])


    df.drop('file', axis=1, inplace=True)


        
    # Define the input layer of the neural network
    input_layer = tf.keras.Input(shape=(224, 224, 3))

    # Define the first convolutional layer
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(conv1)

    # Define the max pooling layer
    max_pool = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    # Define the flattening layer
    flatten = tf.keras.layers.Flatten()(max_pool)

    # Define the first dense layer
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)

    # Define the output layer of the neural network
    output_layer = tf.keras.layers.Dense(len(df['label'].unique()), activation='softmax')(dense1)

    # Create a model object using the defined layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    

    print(df['image'].head())

    X_train, X_val, y_train, y_val = train_test_split(df['image'], df['label'], test_size=0.2, random_state=42) 
    
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])   
    
    model.fit(X_train, y_train, epochs=2, verbose=1, validation_data=(X_val, y_val))
    
    model.save('model.h5')



if __name__ == '__main__':
    main()