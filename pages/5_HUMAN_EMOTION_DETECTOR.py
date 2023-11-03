import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
st.set_page_config(
    page_title="HUMAN EMOTIONS DETECTOR Web App"
)

st.title("HUMAN EMOTIONS DETECTOR")
st.error("This model is taking too much time to predict due to big size but is predicting correctly ")
from tensorflow.keras.layers import Input,Resizing,Rescaling,Permute,Dense
resize_layers = tf.keras.Sequential([
    Resizing(224 ,224),
    Rescaling(1.0/255),
    Permute((3,1,2))
])


import tensorflow as tf
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, TFViTModel

# Define your custom layer registration
def register_custom_layers():
    class CustomTFViTMainLayer(TFViTModel):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)

    tf.keras.utils.get_custom_objects()['TFViTMainLayer'] = CustomTFViTMainLayer

# Call the registration function
register_custom_layers()

# Load your model
model = tf.keras.models.load_model('models/human_emotions_vit.h5')
CLASS_NAMES = ['angry','happy','sad']


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Display the original image
    st.image(image, caption="Original Image")

    # Check if the image is in color (3 channels) and convert to grayscale if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        grayscale_image = image_np

if st.button("PREDICT"):
    image = Image.open(uploaded_image)
    
    # Resize the image to (224, 224)
    image = image.resize((256, 256))

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Perform any other preprocessing steps, if needed

    # Make predictions using your model
    predictions = model.predict(np.expand_dims(image_np, axis=0))
    predicted_class = np.argmax(predictions)
    result = CLASS_NAMES[predicted_class]
    if result == "angry":
        st.header("ANGRY")
    elif result == "sad":
        st.header("SAD")
    elif result == "happy":
        st.header("HAPPY")
if st.button("SHOW DEMO"):
    st.video("videos/human_emotions_model.mp4")



if st.button("KNOW ABOUT MODEL"):
    st.write("""
The architecture described is a deep learning model designed to classify human emotions (happy, angry, or sad) based on input images. This model leverages the power of the ViT (Vision Transformer) architecture and is deployed using Streamlit for user interaction.

1. **Hugging Face Model**:
   - You start by defining the Hugging Face model `'google/vit-base-patch16-224-in21k'`. This is a pre-trained ViT model by Google, which is capable of processing image data for various tasks.

2. **Custom Configuration**:
   - Next, you create a custom ViT configuration by specifying a hidden size of 144. This allows you to fine-tune the model according to your specific requirements.

3. **ViT Model**:
   - You initialize a ViT model using the custom configuration. This ViT model is designed to handle image data and extract meaningful features.

4. **Data Preprocessing Layers**:
   - Before feeding the image data into the ViT model, you set up a series of data preprocessing layers. These layers include:
      - **Resizing**: Resizes the input image to a standard size of 224x224 pixels.
      - **Rescaling**: Normalizes the pixel values to the range [0, 1].
      - **Permute**: Reorders the dimensions of the data to match the model's expected input format.

5. **Feature Extraction**:
   - You load a pre-trained ViT model using the Hugging Face library. This model has been pre-trained on a large dataset and can extract rich features from input images.

6. **Input Layer**:
   - You define an input layer with a shape of (256, 256, 3), indicating that the model expects RGB images with a resolution of 256x256 pixels and 3 color channels.

7. **Model Forward Pass**:
   - You pass the preprocessed input data through the ViT model by applying the resizing and other preprocessing layers. The ViT model extracts features from the image data.

8. **Output Layer**:
   - After processing the image through the ViT model, you extract the relevant features from the model's output. In this case, you take the first position of the output ([:, 0, :]) and connect it to a dense layer with 3 units, using a softmax activation function. This final dense layer is responsible for classifying the emotions into one of the three categories: happy, angry, or sad.

9. **Keras Model**:
   - You create a Keras model that takes the preprocessed input image and produces an output prediction for the emotion classification task.

10. **Deployment with Streamlit**:
   - After defining the architecture of your model, you can deploy it using Streamlit. Streamlit is a Python library for creating interactive web applications with ease. You can use Streamlit to build a user-friendly interface where users can upload images, and your model will classify the emotions of the people in those images in real-time.

In summary, this architecture combines the power of ViT models for image feature extraction with the flexibility of Keras for building the classification head. It allows you to create an interactive application to classify human emotions based on input images, making it accessible and user-friendly.


""")