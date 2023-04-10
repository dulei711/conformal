import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt


def conformal_mapping(img, func):
    """Transforms the input image using a conformal mapping with the given complex function."""
    # Get the size of the input image
    rows, cols, _ = img.shape

    # Compute the inverse mapping
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    z = x + y * 1j
    w = func(z)

    # Convert the complex coordinates to image coordinates
    u = np.real(w).astype(np.float32)
    v = np.imag(w).astype(np.float32)

    # Apply the mapping to the input image
    remapped = cv2.remap(img, u, v, cv2.INTER_LINEAR)

    return remapped


# Define the available complex functions
functions = {
    'Identity': lambda z: z,
    'Sine': np.sin,
    'Cosine': np.cos,
    'Exponential': np.exp,
    'Logarithm': np.log,
}



# Create the Streamlit app
st.title('Conformal Mapping Photo Transformer')

# Upload an image or use the default one
image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if image_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if st.button("run"):

        # Display the original image
        st.subheader('Original Image')
        st.image(image, channels='BGR')

        # Choose a complex function
        function_name = st.selectbox('Select a complex function', list(functions.keys()))

        # Apply the chosen function to the image
        remapped_image = conformal_mapping(image, functions[function_name])

        # Display the transformed image
        st.subheader('Transformed Image')
        st.image(remapped_image, channels='BGR')

else:
    # Read the default image
    st.alert("no image uploaded")
