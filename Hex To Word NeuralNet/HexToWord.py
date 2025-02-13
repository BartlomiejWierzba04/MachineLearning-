import numpy as np
import webcolors
import tensorflow as tf
from colors import colors_dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Generate dataset: hex codes to color names
# This is infact just pulling from the database that i already put into python.
def create_color_dataset():
    colors = colors_dict
    hex_values = []
    color_names = []

    for name, hex in colors.items():
        hex_values.append(hex.lstrip('#'))
        color_names.append(name)
    
    return color_names,hex_values

# Preprocess hex codes to normalized RGB arrays
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return np.array([
        int(hex_code[0:2], 16)/255.0,  # R
        int(hex_code[2:4], 16)/255.0,  # G
        int(hex_code[4:6], 16)/255.0   # B
    ])


        # Prepare data
hex_data, color_labels, = create_color_dataset()
X = np.array([hex_to_rgb(code) for code in hex_data])
y = color_labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
#X_train, X_test, y_train, y_test = train_test_split(
#X, y_encoded, test_size=0.2, random_state=42)

# Build model

def modelBuilding():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(X_train, y_train, 
              epochs=100, 
              batch_size=32,
              validation_split=0.2)

    return model

# Evaluate
#model = modelBuilding()
#loss, accuracy = model.evaluate(X_test, y_test)
#print(f"Test accuracy: {accuracy:.2%}")

#model.save('model_v1.h5')

# Prediction function
def predict_color(hex_code,model):
    rgb = hex_to_rgb(hex_code).reshape(1, -1)
    prediction = model.predict(rgb)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]





