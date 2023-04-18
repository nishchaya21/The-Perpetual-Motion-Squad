import streamlit as st
import os
from PIL import Image
import numpy as np
from main_pred import main_pred

# Define the path where the uploaded images will be saved
UPLOAD_PATH = 'uploads'

# Load the pre-trained machine learning model

# Define a function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((640, 640))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# Define the Streamlit app
def appp():
    # Set the app title
    st.title("InSure")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    # If an image is uploaded, save it to the specified folder and display it
    if uploaded_file is not None:
        # Create the uploads folder if it does not exist
        if not os.path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH)

        # Save the uploaded image to the uploads folder
        image_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the saved image
        image = Image.open(image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and make a prediction
        # processed_image = preprocess_image(image)
        prediction ,checker= main_pred(image_path)
        OMAGE_PATH = 'output.jpg'
        image=Image.open(OMAGE_PATH)
        # Print the prediction
        st.write(f"Prediction: {prediction}")
        if(checker=="car"):
            st.image(image,caption="generated image")
if __name__=="__main__":
    appp()
