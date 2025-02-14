import HexToWord
#import numpy as np
#import webcolors
import tensorflow as tf
import keyboard
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout



model = tf.keras.models.load_model('model_v1.h5')



print("Press the UP arrow key to exit...")

while True:
    hexColor = input("Enter the Hex Colour to get the name") #this can be made into a sanitised input box later - for testing purposes this is fine.
    print(HexToWord.predict_color(hexColor,model))
