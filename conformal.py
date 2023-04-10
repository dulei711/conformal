import streamlit as st
import numpy as np
from sympy import *
from PIL import Image
import matplotlib.pyplot as plt

def square(z):
    return z**2

def sine(z):
    return sin(z)

def logarithm(z):
    return log(z)

def transform_image(image, func):
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Get the height and width of the image
    height, width = img_array.shape[:2]
    
    # Create a meshgrid of complex numbers that corresponds to the image pixels
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    z = x + 1j*y
    
    # Apply the complex function to the meshgrid
    w = func(z)
    
    # Convert the complex numbers back to x and y coordinates
    x_new = np.real(w)
    y_new = np.imag(w)
    
    # Map the x and y coordinates to the original image size
    x_new = ((x_new + 1) / 2) * width
    y_new = ((y_new + 1) / 2) * height
    
    # Interpolate the image to get the transformed image
    transformed_image = np.zeros_like(img_array)
    for i in range(3):
        transformed_image[:,:,i] = np.interp(x_new.flatten(), np.arange(width), img_array[:,:,i].flatten()).reshape((height, width))
    
    # Convert the numpy array back to an image
    transformed_image = Image.fromarray(transformed_image.astype('uint8'))
    
    return transformed_image

st.title('Conformal Mapping Image Transformation')

# Upload the image
image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# Select the conformal mapping function
func_options = {'Square': square, 'Sine': sine, 'Logarithm': logarithm}
func_name = st.selectbox('Select a conformal mapping function', list(func_options.keys()))
func = func_options[func_name]

if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    
    # Transform the image
    transformed_image = transform_image(image, func)
    
    # Display the original and transformed images
    st.image([image, transformed_image

