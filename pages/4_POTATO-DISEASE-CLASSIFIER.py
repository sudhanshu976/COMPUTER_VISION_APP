import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
st.set_page_config(
    page_title="Malaria Detection Web App"
)

st.title("POTATO DISEASE CLASSIFIER")

# Load the model and compile it
potato_model = tf.keras.models.load_model("models/potato_model.h5", compile=False)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)
    image = Image.open(uploaded_image)

if st.button("PREDICT"):

    
    resized = image.resize((256, 256))
    data = np.asarray(resized)  
    img_array = tf.expand_dims(data,0)      
    predictions = potato_model.predict(img_array)
    result=np.argmax(predictions[0])

    if result == 0:
        st.header(' This potato has a Early Blight Disease ')
    elif result == 1:
        st.header('This potato has a Late Blight Disease ')
    else:
        st.header("This potato is Healthy")

if st.button("SHOW DEMO"):
    st.video("videos/potato_model.mp4")



if st.button("KNOW ABOUT MODEL"):
     st.write("""
## Potato-Disease Classifier CNN App Summary

**Model Architecture:**
- Model Type: Convolutional Neural Network (CNN)
- Number of Layers: 6 Convolutional Layers, 6 Max Pooling Layers, 2 Fully Connected Layers , Data Augmentation Layer and Data Preprocessing Layer

**Input:**
- Input Shape: (256, 256, 3)

**Training:**
- Training Data: Plant Village images dataset
- Optimizer: Adam
- Loss Function:  Categorical-Cross-Entropy
- Metrics: Accuracy
- Number of Epochs: 100
- Batch Size: 32

**Model Evaluation:**
- Validation Data: Separate dataset of potato leaf images
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

**Results:**
- Accuracy: 98%

**Conclusion:**
- The CNN model achieved an accuracy of 98% in detecting which potato is healthy , which is sufferiing from early blight disease and which is suffering from late blight disease, indicating strong performance.

**Deployment:**
- Deployment Platform: Web application
- User Interface: Users upload potato leaf images to the app, and it provides a detection of whether a potato is healthy , having late blight or early blight disease.

""")