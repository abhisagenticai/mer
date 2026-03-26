import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

FILE_ID = "1rNGW88oHrzToM5wHzFSXk4hIrooSqtWN1lMt4J4kSB2hlTsWt0mqq9yppuye8-JOe"

if not os.path.exists("best_model.keras"):
    gdown.download(id=FILE_ID, output="best_model.keras", quiet=False)

model = tf.keras.models.load_model("best_model.keras")
class_names = np.load("class_names.npy", allow_pickle=True)
def audio_to_img(audio_path, img_path="temp.png"):
    y, sr = librosa.load(audio_path, sr=22050, duration=10)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(3,3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis("off")
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return img_path

st.title("🎵 Bollywood Mood Classifier")
f = st.file_uploader("Upload audio", type=["wav","mp3","au"])

if f:
    with open(f.name, "wb") as out:
        out.write(f.read())

    img_path = audio_to_img(f.name)
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, 0)

    p = model.predict(x, verbose=0)[0]
    idx = np.argmax(p)

    st.success(f"Prediction: {class_names[idx]}")
    st.write(f"Confidence: {p[idx]:.2%}")
    st.write({str(class_names[i]): float(p[i]) for i in range(len(class_names))})
