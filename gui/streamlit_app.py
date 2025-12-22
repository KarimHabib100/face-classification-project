import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="Face Classification System",
    page_icon="👤",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    model_path = f'models/{model_name}/{model_name}_final.h5'
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open('data/processed/face_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['class_names']
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

def preprocess_image(image, target_size=(160, 160)):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image_resized = cv2.resize(image, target_size)
    # Keep in [0, 255] range - model's preprocess_input handles normalization
    image_normalized = image_resized.astype('float32')

    return image_normalized, image_resized

def predict_image(model, image):
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch, verbose=0)[0]
    return predictions

def main():
    st.markdown('<h1 class="main-header">👤 Face Classification System</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Deep Learning-Based Face Recognition")
    st.markdown("---")
    
    class_names = load_class_names()
    
    if class_names is None:
        st.error("❌ Failed to load class names. Please check data file.")
        return
    
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### Select Model")
        model_choice = st.selectbox(
            "Choose CNN Architecture:",
            ["ResNet50", "InceptionV3", "EfficientNetB0"]
        )
        
        model_name = model_choice.lower().replace(" ", "")
        
        st.markdown("---")
        
        st.markdown("### Input Method")
        input_method = st.radio(
            "Choose input source:",
            ["📁 Upload Image", "📷 Webcam Capture"]
        )
        
        st.markdown("---")
        
        with st.expander("ℹ️ About"):
            st.markdown(f"""
            **Models:** ResNet50, InceptionV3, EfficientNetB0
            
            **Dataset:** Labeled Faces in the Wild (LFW)
            
            **Classes:** {len(class_names)} people
            """)
    
    model = load_model(model_name)
    
    if model is None:
        st.warning(f"⚠️ Please check if {model_choice} model exists.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Model", model_choice)
    with col2:
        st.metric("Number of Classes", len(class_names))
    with col3:
        st.metric("Model Parameters", f"{model.count_params():,}")
    
    st.markdown("---")
    
    image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a face image...",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    else:
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            image = Image.open(picture)
            image = np.array(image)
    
    if image is not None:
        processed_image, display_image = preprocess_image(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Input Image")
            st.image(display_image, use_column_width=True)
        
        if st.button("🔍 Classify Face", type="primary"):
            with st.spinner("Analyzing..."):
                predictions = predict_image(model, processed_image)
                
                with col2:
                    st.markdown("### 🎯 Top Predictions")
                    
                    top_indices = np.argsort(predictions)[-3:][::-1]
                    
                    for i, idx in enumerate(top_indices, 1):
                        confidence = predictions[idx]
                        class_name = class_names[idx]
                        
                        st.markdown(f"**{i}. {class_name}**")
                        st.progress(float(confidence))
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        st.markdown("---")
                
                st.markdown("---")
                st.markdown("### 📊 All Predictions")
                
                top_10_indices = np.argsort(predictions)[-10:][::-1]
                
                chart_data = {}
                for idx in top_10_indices:
                    chart_data[class_names[idx]] = predictions[idx]
                
                st.bar_chart(chart_data)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Face Classification System | Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()