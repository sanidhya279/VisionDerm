# VisionDerm
# VisionDerm 🩺

VisionDerm is a deep learning web application that detects and classifies common skin conditions from images. 

## About the Project
This project uses a fine-tuned EfficientNet B4 model built with PyTorch. It analyzes uploaded images and predicts the probability of five specific skin conditions:
* Acne
* Eczema
* Pigmentation
* Psoriasis
* Rosacea

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Web Interface:** Streamlit
* **Model Architecture:** EfficientNet B4
* **Image Processing:** Pillow

## Live Demo
[Insert your Streamlit app link here!]

## How to Run Locally
If you want to run this project on your own computer, follow these steps:

1. Clone this repository.
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Run the Streamlit app:
   `streamlit run app.py`

*(Note: The pre-trained model file is large and will automatically download from Google Drive the first time you run the app.)*
