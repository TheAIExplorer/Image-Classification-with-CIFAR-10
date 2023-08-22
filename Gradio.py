#!/usr/bin/env python
# coding: utf-8

# In[21]:


import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("Image Classification with CIFAR-10.h5")

# label mapping
labels = '''Airplane Automobile Bird Cat Deer Dog Frog Horse Ship Truck'''.split()

def classify_image(input_image):
    # Convert Gradio image to numpy array
    image = np.array(input_image)
    
    # Resize the image to 32x32 pixels
    resized_image = Image.fromarray(image).resize((32, 32))
    
    # Convert the resized image to an array
    image_array = np.array(resized_image)
    
    # Expand dimensions to match the model input shape
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make predictions using the model
    predicted_label = labels[model.predict(image_array).argmax()]
    
    return f"{predicted_label}"

input_image = gr.inputs.Image()  # No need to specify shape
output_text = gr.outputs.Textbox()

# Create the Gradio interface
demo = gr.Interface(fn=classify_image, inputs=input_image, outputs=output_text,
                     title="CNN Image Classifier", 
                     description="Upload an image and get the predicted class out of Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.")
demo.launch()


# In[ ]:




