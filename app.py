import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Clean Page Configuration
st.set_page_config(page_title="Waste Classification Analysis", layout="centered")

st.title("Waste Material Classification")
st.markdown("Upload an image of the material to run the classification model.")

# Load Model (Silently)
@st.cache_resource(show_spinner=False)
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# File Uploader
uploaded_file = st.file_uploader("Select an image file (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2 column layout
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
        
    with col2:
        st.subheader("Analysis Results")
        
        # Preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Calculate primary prediction
        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index]
        
        # Clean data display
        st.markdown(f"**Predicted Material:** {class_names[predicted_index]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        
        st.divider()
        st.markdown("**Probability Breakdown:**")
        
        # Simple text-based breakdown rather than a heavy chart
        for i, prob in enumerate(output_data):
            st.text(f"{class_names[i]}: {prob:.2%}")