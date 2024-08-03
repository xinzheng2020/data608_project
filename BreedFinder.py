import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk
from datetime import datetime
import requests
import base64
import pandas as pd
import concurrent.futures
import os

# Load and display the logo and a dog picture
logo = Image.open('breedfinder_logo.png')
dog_picture = Image.open('dog-breed-identifier.png')
# Load breed features
breed_features = pd.read_csv("breed_features.csv")

# Title and subtitle with custom colors
st.markdown("<h1 style='color: black;'>Identify dog breeds using AI</h1>", unsafe_allow_html=True)

# Resize the images to the desired height while maintaining aspect ratio
desired_height = 200  # Set the desired height in pixels

logo_ratio = desired_height / logo.height
logo = logo.resize((int(logo.width * logo_ratio), desired_height))

sample_picture_ratio = desired_height / dog_picture.height
sample_picture = dog_picture.resize((int(dog_picture.width * sample_picture_ratio), desired_height))

# Create two columns for the logo and the sample picture
col1, col2 = st.columns([1, 1])

# Display the logo in the left column
with col1:
    st.image(logo, use_column_width=False)

# Display the sample picture in the right column
with col2:
    st.image(sample_picture, caption='We are dog lovers!', use_column_width=False)

# Display five most popular breeds
st.markdown("<h2 style='color: green;'>Most Popular Breeds:</h2>", unsafe_allow_html=True)
popular_breeds = [
    ("Labrador Retriever", "Labrador Retriever.jpg"),
    ("Corgi", "Corgi.jpg"),
    ("French Bulldog", "French Bulldog.jpg"),
    ("German Shepherd", "German Shepherd.jpg"),
    ("Golden Retriever", "Golden Retriever.jpg")
]

cols = st.columns(5)
for col, breed in zip(cols, popular_breeds):
    breed_name, breed_image = breed
    image = Image.open(breed_image)
    col.image(image, caption=breed_name, use_column_width=True)

# Streamlit UI
st.title("Dog Breed Identifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load your trained TensorFlow model
@st.cache_resource
def load_model():
    return tfk.models.load_model('20240728-19371722217023-full-image-set-mobilenetv2-Adam.h5',
                                 custom_objects={"KerasLayer": hub.KerasLayer})

model = load_model()

labels_csv = pd.read_csv("labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)

# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data is a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  # filepaths
                                                   tf.constant(y)))  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                   tf.constant(y)))
        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
    return data_batch

# Define image size
IMG_SIZE = 224

# Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns the image into a Tensor.
    """
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired value (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image

# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# Function to handle image prediction
def handle_prediction(uploaded_file):
    image = Image.open(uploaded_file)
    # Create a folder named by the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_path = os.path.join("uploaded_images", timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded image in the timestamped folder
    file_path = os.path.join(folder_path, "1.jpg")
    image.save(file_path)
    path = [file_path]
    encoded_image = base64.b64encode(uploaded_file.read()).decode('utf-8')
    custom_data = create_data_batches(path, test_data=True)
    custom_preds = model.predict(custom_data)
    response_time = datetime.now() - start_time
    pred_breed = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    confidence = np.max(custom_preds)
    return encoded_image, pred_breed, confidence, response_time, file_path

def send_data_to_api(api_url, data):
    response = requests.put(api_url, json=data)
    return response

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Identifying the breed...")

    start_time = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(handle_prediction, uploaded_file)
        encoded_image, pred_breed, confidence, response_time, file_path = future.result()

    # Get the breed features
    breed_info = breed_features[breed_features['breed'] == pred_breed[0]].iloc[0]

    name_display = breed_features[breed_features['breed'] == pred_breed[0]].iloc[0]['Breed_name']
    # Display prediction results in a styled frame
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9; display: flex; justify-content: space-between;">
            <div>
                <h3 style="color: #4CAF50;">Prediction Results</h3>
                <p><strong>Predicted Breed:</strong> {name_display}</p>
                <p><strong>Confidence:</strong> {(confidence*100):.1f}%</p>
                <p><strong>Response Time:</strong> {response_time.total_seconds()} seconds</p>
            </div>
            <div style="margin-left: 20px;">
                <h3 style="color: #4CAF50;">Breed Information</h3>
                <p><strong>Origin:</strong> {breed_info['Origin']}</p>
                <p><strong>Size:</strong> {breed_info['Size']}</p>
                <p><strong>Personality:</strong> {breed_info['Personality']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2 style='color: red;'>Is the prediction correct?</h2>", unsafe_allow_html=True)
    feedback = st.radio("", ["Yes", "No"], index=None, horizontal=True)
    true_label = pred_breed if feedback == "Yes" else ""
    st.markdown("<h2 style='color: green;'>Please rate our service:</h2>", unsafe_allow_html=True)
    rating = st.slider("", 1, 5, 3)

    if feedback == "No":
        true_label = st.text_input("Please enter the correct breed:")

    if st.button("Submit"):
        api_url = "https://zyz1e1zka7.execute-api.us-east-1.amazonaws.com/dev/update"
        data = {
            "picture_path": encoded_image,
            "prediction_result": pred_breed,
            "true_label": true_label,
            "rating": rating
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(send_data_to_api, api_url, data)
            response = future.result()

        if response.status_code == 200:
            st.success("Data successfully written to DynamoDB!")
        else:
            st.error(f"Failed to write data. Status code: {response.status_code}")
            st.json(response.json())
        
        # Clear previous questions
        st.session_state["feedback"] = None
        st.markdown("<h2 style='color: blue;'>Thank you for giving us feedback! 🐶❤️</h2>", unsafe_allow_html=True)
