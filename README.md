# data608_project

# BreedFinder: Dog Breed Identification using AI

## Project Overview

BreedFinder is an AI-powered web application designed to identify dog breeds from user-uploaded images. By leveraging a Convolutional Neural Network (CNN) trained on a large dataset of labeled dog images, this application provides accurate breed predictions along with confidence scores and additional breed information. The system is built using TensorFlow, Keras, and Streamlit, and is deployed using Docker and AWS services.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Acknowledgements](#acknowledgements)

## Features

- Upload an image of a dog to identify its breed.
- Provides breed predictions with confidence scores.
- Displays additional breed information.
- Allows users to provide feedback on prediction accuracy.
- Stores user feedback and images for continuous model improvement.
- Interactive and user-friendly interface built with Streamlit.
- Deployed using Docker and hosted on AWS EC2.

## Setup and Installation

### Prerequisites

- Docker
- Git

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/BreedFinder.git
   cd BreedFinder

2. **Build the Docker image:**
   bash:
   docker build -t breedfinder .
   
3. **Run the Docker container:**
   bash:
   docker run -p 8502:8502 breedfinder

## Usage

Open your web browser and go to http://localhost:8502.

Upload an image of a dog using the file uploader.

View the predicted breed, confidence score, and additional breed information.

Provide feedback on the prediction accuracy.

Rate the service and submit your feedback.

## Files

- Corgi.jpg, French Bulldog.jpg, German Shepherd.jpg, Golden Retriever.jpg, Labrador Retriever.jpg: Sample images of popular dog breeds.

- 20240728-19371722217023-full-image-set-mobilenetv2-Adam.h5: Pre-trained TensorFlow model.

- breed_features.csv: CSV file containing additional information about various dog breeds.

- breedfinder_logo.png, dog-breed-identifier.png: Images used in the application interface.

- BreedFinder.py: Main application script built with Streamlit.

- Dockerfile: Instructions to build the Docker image for the application.

- labels.csv: CSV file containing the labels for the dog breed images.

- requirements.txt: List of Python dependencies required for the application.


## Acknowledgements

The dataset used for training the model was sourced from Kaggle's Dog Breed Identification competition.

This project leverages the power of TensorFlow, Keras, and Streamlit for machine learning and web interface development.

Special thanks to the open-source community for providing valuable resources and tools.
