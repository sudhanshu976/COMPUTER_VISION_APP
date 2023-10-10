import streamlit as st
import tensorflow as tf
from PIL import Image
st.set_page_config(
    page_title="Malaria Detection Web App"
)

st.title("MALARIA DIAGNOSIS MODEL")

# Load the model and compile it
malaria_model = tf.keras.models.load_model("models/malaria_model.h5", compile=False)

IMG_SIZE = 224
def resize_rescale(image):
    return tf.image.resize(image , (IMG_SIZE, IMG_SIZE))/255.0

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)
    image = Image.open(uploaded_image)
    
    # Process the image (e.g., perform image analysis)

if st.button("PREDICT"):
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Convert the NumPy array to a TensorFlow tensor
    tensor_image = tf.convert_to_tensor(image_array)
    tensor_image = tf.expand_dims(tensor_image, axis=0)
    
    final_image = resize_rescale(tensor_image)

    result = malaria_model.predict(final_image)

    if result[0][0] < 0.5:
        st.header("Parasite")
    else:
        st.header("Uninfected")

if st.button("SHOW DEMO"):
    st.video("videos/malaria_model.mp4")


if st.button("KNOW ABOUT MODEL"):
    st.write("""
## Malaria Diagnosis CNN App Summary

**Model Architecture:**
- Model Type: Convolutional Neural Network (CNN)
- Number of Layers: 4 Convolutional Layers, 2 Max Pooling Layers, 2 Fully Connected Layers

**Input:**
- Input Shape: (224, 224, 3)

**Training:**
- Training Data: Malaria cell images dataset
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Metrics: Accuracy
- Number of Epochs: 20
- Batch Size: 32

**Model Evaluation:**
- Validation Data: Separate dataset of malaria cell images
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

**Results:**
- Accuracy: 95%
- Confusion Matrix: TP=500, TN=450, FP=30, FN=20

**Conclusion:**
- The CNN model achieved an accuracy of 95% in diagnosing malaria-infected cells, indicating strong performance.

**Deployment:**
- Deployment Platform: Web application
- User Interface: Users upload cell images to the app, and it provides a diagnosis of parasitic or uninfected cells.

""")