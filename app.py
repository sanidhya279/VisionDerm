import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# 1. Setup Class Names and Model Path
CLASS_NAMES = ['Acne', 'Eczema', 'Pigmentation', 'Psoriasis', 'Rosacea']
MODEL_PATH = "visionderm_model.pth"
FILE_ID = "1ojNmrZ326m4ecr5BYfNI1huNE2KwRtuG"

# 2. Image Transformations
data_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load the Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... this might take a minute or two."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
            
    # Load empty skeleton of EfficientNet B4
    model = models.efficientnet_b4(weights=None)
    
    # Adjust final layer for your 5 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    
    # Load your trained weights safely on CPU
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    return model

# Initialize the model
model = load_model()

# 4. Streamlit Web Interface
st.title("VisionDerm")
st.subheader("Deep Learning Based Detection of Skin Diseases")
st.write("Upload an image of a skin condition to get a prediction.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    
    st.write("Analyzing...")
    
    # Preprocess the image
    input_tensor = data_transform(image).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        
    # Get the final label and percentage
    final_prediction = CLASS_NAMES[predicted_class_idx.item()]
    final_confidence = confidence.item() * 100
    
    # Display the result
    st.success(f"**Prediction:** {final_prediction}")
    st.info(f"**Confidence:** {final_confidence:.2f}%")
