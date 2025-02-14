import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("lung.h5")

map_dict = {
    0: 'Adenocarcinoma',
    1: 'Carcinoma de grandes células',
    2: 'Normal',
    3: 'Carcinoma de células escamosas'
}

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (200, 200))
    resized = resized.astype('float32') / 255.0
    st.image(opencv_image, channels="RGB", caption="Imagem carregada")
    img_reshape = np.expand_dims(resized, axis=0)
    if st.button("Gerar Previsão"):
        prediction = model.predict(img_reshape).argmax()
        st.write(f"**A classe prevista para a imagem é:** {map_dict[prediction]}")
