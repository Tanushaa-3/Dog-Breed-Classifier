

# streamlit_app.py - Your main Streamlit application

import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tempfile
import os

# Try to import tensorflow with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    st.error(f"❌ TensorFlow import failed: {e}")
    st.info("Please check your requirements.txt and Python version compatibility")

# Page configuration
st.set_page_config(
    page_title="🐕 AI Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading to make it faster
@st.cache_resource
def load_model_and_breeds():
    """Load model and breeds list (cached for performance)"""
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not available. Cannot load model.")
        return None, None
        
    try:
        # Check if model file exists
        if not os.path.exists('dog_classifier.h5'):
            st.error("❌ Model file 'dog_classifier.h5' not found. Please upload the model file.")
            return None, None
            
        if not os.path.exists('unique_breeds.pkl'):
            st.error("❌ Breeds file 'unique_breeds.pkl' not found. Please upload the breeds file.")
            return None, None
            
        # Load the trained model
        model = tf.keras.models.load_model('dog_classifier.h5')
        
        # Load unique breeds
        with open('unique_breeds.pkl', 'rb') as f:
            unique_breeds = pickle.load(f)
            
        return model, unique_breeds
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure both 'dog_classifier.h5' and 'unique_breeds.pkl' are in your repository root")
        return None, None

# Preprocessing functions (same as your Colab)
def preprocess_image(image_path, img_size=224):
    """Preprocess image exactly like in training"""
    if not TF_AVAILABLE:
        return None
        
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[img_size, img_size])
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def get_image_label(image_path, label):
    """Get image and label pair"""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, label
    return processed_image, label

def create_data_batches(x, y=None, batch_size=32, valid_data=False, test_data=False):
    """Create data batches - same as your training code"""
    if not TF_AVAILABLE:
        return None
        
    try:
        if test_data:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
            data_batch = data.map(preprocess_image).batch(batch_size)
            return data_batch
        elif valid_data:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
            data_batch = data.map(get_image_label).batch(batch_size)
            return data_batch
        else:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
            data = data.shuffle(buffer_size=len(x))
            data = data.map(get_image_label)
            data_batch = data.batch(batch_size)
        return data_batch
    except Exception as e:
        st.error(f"Error creating data batches: {e}")
        return None

def get_pred_label(prediction_prob, unique_breeds):
    """Turn prediction probabilities into breed labels"""
    try:
        return unique_breeds[np.argmax(prediction_prob)]
    except Exception as e:
        st.error(f"Error getting prediction label: {e}")
        return None

def predict_dog_breed(image, model, unique_breeds):
    """Predict dog breed using the exact same pipeline as training"""
    if not TF_AVAILABLE or model is None:
        return None, 0, []
        
    temp_image_path = None
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, format='JPEG')
            temp_image_path = tmp_file.name
        
        # Use exact same preprocessing pipeline
        temp_data = create_data_batches([temp_image_path], test_data=True)
        
        if temp_data is None:
            return None, 0, []
        
        # Make prediction
        predictions = model.predict(temp_data, verbose=0)
        
        # Get predicted breed
        predicted_breed = get_pred_label(predictions[0], unique_breeds)
        confidence = np.max(predictions[0]) * 100
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = []
        
        for idx in top_5_indices:
            breed_name = unique_breeds[idx].replace('_', ' ').title()
            prob = predictions[0][idx] * 100
            top_predictions.append((breed_name, prob))
        
        return predicted_breed, confidence, top_predictions
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, 0, []
    finally:
        # Clean up temp file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass

# Main Streamlit App
def main():
    # Title and description
    st.title("🐕 AI Dog Breed Classifier")
    st.markdown("---")
    
    # Check TensorFlow availability first
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not available. Please check your environment setup.")
        st.markdown("""
        ### 🔧 Troubleshooting Steps:
        1. **Check your requirements.txt** - Make sure it contains:
           ```
           streamlit>=1.28.0
           tensorflow>=2.12.0,<2.16.0
           pillow>=9.5.0
           numpy>=1.24.0,<2.0.0
           ```
        
        2. **Python Version** - TensorFlow 2.12+ requires Python 3.8-3.11
        
        3. **Streamlit Cloud** - Try using these versions:
           ```
           streamlit==1.32.0
           tensorflow==2.15.0
           pillow==10.0.0
           numpy==1.24.3
           ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        **🔥 Features:**
        - Identifies 120+ dog breeds
        - High accuracy AI model
        - Instant predictions
        - Top 5 breed suggestions
        
        **🚀 How to use:**
        1. Upload a clear dog photo
        2. Wait for AI analysis
        3. See breed prediction & confidence
        
        **💡 Tips:**
        - Use clear, well-lit photos
        - Single dog works best
        - Close-up shots are ideal
        """)
        
        st.markdown("---")
        st.markdown("**Built with:**")
        st.markdown("🧠 TensorFlow • 🎯 Streamlit • 🤖 AI/ML")
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, unique_breeds = load_model_and_breeds()
    
    if model is None or unique_breeds is None:
        st.error("❌ Could not load the AI model. Please check if model files are uploaded correctly.")
        st.markdown("""
        ### 📁 Required Files:
        Make sure these files are in your repository root:
        - `dog_classifier.h5` (your trained model)
        - `unique_breeds.pkl` (list of dog breeds)
        - `streamlit_app.py` (this file)
        - `requirements.txt` (dependencies)
        """)
        return
    
    # Success message
    st.success(f"✅ AI Model loaded successfully! Ready to identify {len(unique_breeds)} dog breeds.")
    
    # File uploader
    st.subheader("📸 Upload Dog Image")
    uploaded_file = st.file_uploader(
        "Choose a dog image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of a dog for breed identification"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            st.subheader("📷 Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your uploaded dog photo", use_column_width=True)
                
                # Image info
                st.info(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
        
        with col2:
            # Prediction button
            if st.button("🔍 Identify Breed", type="primary"):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Make prediction
                    predicted_breed, confidence, top_predictions = predict_dog_breed(
                        image, model, unique_breeds
                    )
                
                if predicted_breed and top_predictions:
                    # Main prediction
                    st.subheader("🎯 Prediction Result")
                    formatted_breed = predicted_breed.replace('_', ' ').title()
                    
                    # Confidence color coding
                    if confidence > 80:
                        confidence_color = "🟢"
                    elif confidence > 60:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"
                    
                    st.success(f"**Primary Prediction:** {formatted_breed}")
                    st.metric("Confidence", f"{confidence:.1f}%", delta=f"{confidence_color}")
                    
                    # Top 5 predictions
                    st.subheader("🏆 Top 5 Predictions")
                    
                    for i, (breed, prob) in enumerate(top_predictions):
                        col_rank, col_breed, col_prob = st.columns([0.5, 2, 1])
                        
                        with col_rank:
                            if i == 0:
                                st.markdown("🥇")
                            elif i == 1:
                                st.markdown("🥈")
                            elif i == 2:
                                st.markdown("🥉")
                            else:
                                st.markdown(f"{i+1}.")
                        
                        with col_breed:
                            st.markdown(f"**{breed}**")
                        
                        with col_prob:
                            st.markdown(f"{prob:.1f}%")
                    
                    # Progress bars for visual representation
                    st.subheader("📊 Confidence Visualization")
                    for breed, prob in top_predictions:
                        st.progress(prob/100, text=f"{breed}: {prob:.1f}%")
                
                else:
                    st.error("❌ Could not process the image. Please try another image.")
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload a dog image to get started!")
        
        # Example images section
        st.subheader("📚 What You'll Get")
        st.markdown("""
        **Your results will include:**
        - 🎯 Primary breed prediction
        - 📊 Confidence percentage  
        - 🏆 Top 5 possible breeds
        - 📈 Visual confidence bars
        """)

# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🐕 AI Dog Breed Classifier | Built with ❤️ using Streamlit & TensorFlow</p>
        <p>Made for dog lovers, by dog lovers! 🐾</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()

       
      
