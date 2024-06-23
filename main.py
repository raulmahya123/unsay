import streamlit as st
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Flatten, Dropout, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
import numpy as np

# Categorical focal loss
def categorical_focal_loss(y_true, y_pred, gamma=2.5, alpha=0.5):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss

# Function to define a multi-input functional model
def get_model_functional(input_size, filters, kernel_size, regularizerL2):
    def conv_block(x, filter_size):
        x = Conv2D(filter_size, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(regularizerL2))(x)
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)
        x = relu(x)
        return x

    # Network 1
    input_n1 = Input(shape=(input_size, input_size, 1))
    x1 = conv_block(input_n1, filters[0])
    x1 = conv_block(x1, filters[1])
    x1 = conv_block(x1, filters[2])
    flat_n1 = Flatten()(x1)

    # Network 2
    input_n2 = Input(shape=(input_size, input_size, 1))
    x2 = conv_block(input_n2, filters[0])
    flat_n2 = Flatten()(x2)

    # Network 3
    input_n3 = Input(shape=(input_size, input_size, 1))
    x3 = conv_block(input_n3, filters[0])
    x3 = conv_block(x3, filters[1])
    flat_n3 = Flatten()(x3)

    # Merge all networks
    merged = concatenate([flat_n1, flat_n2, flat_n3])
    merged = Dropout(0.5)(merged)
    output = Dense(NUM_CLASSES, activation='softmax')(merged)

    return [input_n1, input_n2, input_n3], output

# Parameters
PATCH_SIZE = 64
NUM_CLASSES = 5
filters_size = [64, 128, 256]
kernel_size = 3
regularizerL2 = 0.0005

# Build the model
inputs, output = get_model_functional(PATCH_SIZE, filters_size, kernel_size, regularizerL2)
model = Model(inputs=inputs, outputs=output)

# Compile the model
optimizer_fcn = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer_fcn, loss=categorical_focal_loss, metrics=['accuracy'])

# Function to preprocess an image
def preprocess_image(image_path, target_size=(PATCH_SIZE, PATCH_SIZE)):
    img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title('Multi-input Model Prediction')

# File upload for images
uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        st.write("")
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Preprocess and make prediction
        image = preprocess_image(uploaded_file)
        predictions = model.predict([image, image, image])
        
        # Display predictions
        st.write(f'Predictions: {predictions}')

