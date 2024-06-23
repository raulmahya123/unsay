import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Function to preprocess an image
def preprocess_image(image_path, target_size=(64, 64)):
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Function to load and predict using the model
def load_and_predict(model_path, image_paths):
    # Load the model
    model = load_model(model_path, compile=False)

    # Preprocess images
    images = []
    for image_path in image_paths:
        img_array = preprocess_image(image_path)
        if img_array is not None:
            images.append(img_array)

    # Make predictions
    if images:
        predictions = model.predict(images)
        return predictions
    else:
        return None

# Main function to define the Streamlit app
def main():
    st.title('Multi-Input Functional Model with Streamlit')

    # File upload section
    st.sidebar.title('Upload Image Files')
    uploaded_files = st.sidebar.file_uploader('Choose up to 3 images...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    # Display uploaded images and predictions
    if uploaded_files:
        st.header('Uploaded Images:')
        image_paths = []
        for uploaded_file in uploaded_files:
            st.subheader(f'File: {uploaded_file.name}')
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            image_paths.append(uploaded_file)

        # Button to perform predictions
        if st.sidebar.button('Predict'):
            st.header('Predictions:')
            st.write('Processing...')

            # Perform prediction
            model_path = 'path_to_your_model/model.h5'  # Update with your model path
            predictions = load_and_predict(model_path, image_paths)

            # Display predictions
            if predictions is not None:
                st.write('Predicted probabilities:')
                st.write(predictions)

                # Display predicted class
                predicted_class = np.argmax(predictions, axis=1)
                st.write(f'Predicted class: {predicted_class}')

                # Display bar chart of predictions
                classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # Update with your classes
                plt.bar(classes, predictions[0])
                plt.xlabel('Classes')
                plt.ylabel('Probability')
                plt.title('Predicted Probabilities')
                st.pyplot(plt)
            else:
                st.write('No valid images to predict.')

# Entry point of the Streamlit app
if __name__ == '__main__':
    main()
