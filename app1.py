import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import io
import shutil
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
from keras import models
from keras import layers
from io import BytesIO, StringIO
from PIL import Image, ImageOps
import streamlit as st

model = keras.models.load_model('model.h5')

def main():
    
    page = st.sidebar.selectbox("Choose a page", ["Model description", "Prediction on your file"])
    if page == "Model description":
        st.header("Dogs and Cats model")
        st.write("This model predicts what is shown in the picture: a cat or a dog. If the cat is 0, if the dog is 1.")
        col1, col2, col3 = st.beta_columns(3)
        i=0
        for dirname, _, filenames in os.walk('../deploy/data/pics'):
            for filename in filenames:
                img = plt.imread(os.path.join(dirname, filename))
                if i<4:
                    col1.image(img,use_column_width=True)
                elif i>=4 and i<8:
                    col2.image(img,use_column_width=True)
                else:
                    col3.image(img,use_column_width=True)
                i+=1
    
    
    elif page == "Prediction on your file":
        st.title("Prediction with model on your picture")
        img_data = st.file_uploader("Upload file", type=["jpg", "png"])
        
        if img_data is not None:
            uploaded_image = Image.open(img_data)
            st.image(uploaded_image)

            size = (150, 150)
            image = ImageOps.fit(uploaded_image, size, Image.ANTIALIAS)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]
            pred = model.predict(img_reshape)

            st.title('Final predict is:')
            st.write(pred)
            if pred>0.5:
                st.write('dog')
            else:
                st.write('cat')
        
        

if __name__ == "__main__":
    main()
