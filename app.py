import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


def prediksi_gambar(gambar_data, model):
    nama_class = ['arm wrestling', 'fencing', 'rugby', 'surfing', 'water polo']
    ukuran = (224, 224)
    gambar = ImageOps.fit(gambar_data,ukuran, Image.ANTIALIAS)
    gambar_toarray = tf.keras.utils.img_to_array(gambar)
    gambar_toarray = tf.expand_dims(gambar_toarray, 0) 
    prediksi = model.predict(gambar_toarray)
    presentase = tf.nn.softmax(prediksi[0])
    st.markdown("<p style='text-align: center; font-size:30px; font-weight:bold; color: black;'>Diklasifikasikan Jenis Olahraga {} </h1>"
                .format(nama_class[np.argmax(presentase)]),unsafe_allow_html=True)


st.title("WEB KLASIFIKASI JENIS OLAHRAGA")
file = st.file_uploader("Masukan Gambar Olahraga", type=['jpg'])

if file is None:
    st.text("Gambar Belum Dimasukan")
    
else:
    gambar = Image.open(file)
    st.image(gambar, use_column_width=True)
    model = tf.keras.models.load_model('ModelKlasifikasiOlahraga.h5')
    mulai_prediksi = st.button("KLASIFIKASI")
    
    if mulai_prediksi:
        prediksi_gambar(gambar, model)