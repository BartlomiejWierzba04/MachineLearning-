import HexToWord
import numpy as np
import webcolors
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



model = tf.keras.models.load_model('model_v1.h5')

print(HexToWord.predict_color("#FF00FF",model))
