import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the same CNN model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 classes for EMNIST ByClass

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Function to load the trained model
@st.cache_resource # Cache the model loading
def load_model():
    model = Net()
    # Load on CPU as Streamlit deployment might not have GPU
    model.load_state_dict(torch.load("emnist_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the class mapping for EMNIST ByClass (0-9, A-Z, a-z)
# This mapping needs to be accurate based on the dataset documentation
emnist_classes = [str(i) for i in range(10)] + \
                 [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                 [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Streamlit application structure
st.title("Real-time Character Recognition")

st.write("Upload an image of a character (digit, uppercase, or lowercase letter) for recognition.")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

    # Load the trained model
    model = load_model()

    # Get predictions
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_character = emnist_classes[predicted_class_index]

    # Display the predicted character
    st.success(f"Predicted Character: {predicted_character}")

st.write("Instructions:")
st.write("1. Train the model by running the training script (`train_emnist.py`).")
st.write("2. Make sure the trained model file (`emnist_model.pth`) is in the same directory as this script.")
st.write("3. Save this script as a Python file (e.g., `app.py`).")
st.write("4. Open your terminal or command prompt, navigate to the directory containing the files.")
st.write("5. Run the Streamlit application using the command: `streamlit run app.py`")
st.write("6. Your web browser will open with the character recognition application.")
