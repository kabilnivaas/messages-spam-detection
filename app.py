import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import cv2
import numpy as np
import pytesseract
from nltk.stem.porter import PorterStemmer
from PIL import Image
import io
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Image Spam Classifier")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    img_bytes = img_file_buffer.read()
    image = Image.open(io.BytesIO(img_bytes))
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if image.mode == 'RGB':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        roi = image_np[y:y+h, x:x+w]
        
        text += pytesseract.image_to_string(roi)
    input_sms = text
    if st.button('Predict'):

        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
else:
    st.write("Please upload an image file.")
