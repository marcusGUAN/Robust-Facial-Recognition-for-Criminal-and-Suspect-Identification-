import streamlit as st
import pickle
import cv2
import time
import os
import numpy as np
from keras.models import load_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder

st.title('Criminal Face Recognition')

with st.expander("User Manual: How to Use"):
    st.write("1. Upload a celebrity image. The image can be celebrities wearing masks, glasses, hat etc.")
    st.write("2. Wait for the system to complete running the prediction")
    st.write("3. View the output result with bounding box, recognition result and confidence level")
    st.write("4. Thank you for using the system")

uploaded_image = st.file_uploader("Upload the Image")

#To extract the detected face and pass it into the model
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    face = detector.detect_faces(pixels)
    x1,y1,w,h = face[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    face_output = pixels[y1:y2,x1:x2]
    image = Image.fromarray(face_output,'RGB')
    image = image.resize((160,160))
    face_array = np.asarray(image)
    return face_array

#To acquire the embedding for the input image
def get_embedding(face):
    model = load_model("E:/APU/APU work/FYP/FYP_criminal/FYP_GUI/facenet_keras.h5")
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    image = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    result = model.predict(image)
    return result[0]

def predictor(img):
    #Image processing
    image = extract_face(img)
    image = get_embedding(image)
    image = np.expand_dims(image, axis=0)
    #Loading trained model using pickle library
    model = pickle.load(open(r'.\support_vector_classifier.sav', 'rb'))
    #Inference
    prediction = model.predict(image)
    #Getting the label of predicted image
    encoder_output = LabelEncoder()
    encoder_output.classes_ = np.load('encoded_celeb.npy')
    predicted_name = encoder_output.inverse_transform(prediction)
    #Calculating probability that face matches
    probability = model.predict_proba(image)
    class_probability = probability[0] * 100
    confidence = max(class_probability)

    return predicted_name[0], confidence

#If there is an uploaded image, then load the image and display
if uploaded_image is not None:
    col1, col2 = st.columns(2)

    # display the input image
    input_image = Image.open(uploaded_image)
    input_image.save(uploaded_image.name)

    with col1:
        st.header("Input Image")
        st.image(input_image)
        #run the prediction
        prediction_output = predictor(uploaded_image.name)

        with col2:
            st.header("Output of Result")
            pixels = np.array(input_image)
            #detect face and draw bounding box
            detector = MTCNN()
            detected_face = detector.detect_faces(pixels)
            if len(detected_face) > 0:
                for face in detected_face:
                    x, y, width, height = face['box']
                    x = abs(x)
                    y = abs(y)
                    x2 = abs(x + width)
                    y2 = abs(y + height)
                    cv2.rectangle(pixels, (x, y), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(pixels, prediction_output[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

            #display input image with bounding box
            st.image(pixels)
            #show recognition results
            st.write("Recognised as:", prediction_output[0])
            st.write("Confidence:", prediction_output[1])

    time.sleep(2)
    os.remove(uploaded_image.name)