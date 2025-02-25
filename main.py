import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pytesseract
import mysql.connector
import uuid
from datetime import datetime
from tensorflow.keras.preprocessing import image
from PIL import Image
from huggingface_hub import InferenceClient

# ✅ Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Load trained CNN model
model = tf.keras.models.load_model("logo_classifier_model.h5")
class_labels = ['Original Ford  logo','Original Ferrari logo']

# ✅ Hugging Face API setup
HUGGINGFACE_API_KEY = ""
client = InferenceClient(model="google/gemma-2-27b-it", token=HUGGINGFACE_API_KEY)

# ✅ MySQL Config
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "mourya"  # Change this
MYSQL_DATABASE = "invoice_predictions"

# ✅ Streamlit UI
st.set_page_config(page_title="FORD & FERRARI Invoice Analyzer", layout="wide")
st.title("📄 FORD & FERRARI Invoice Extraction")

# 📤 File Upload Section
uploaded_file = st.file_uploader("Upload an Invoice Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # 🔹 Save uploaded image temporarily
    img_path = "temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Preprocess Image & Predict
    def predict_logo(img_path):
        img_size = (224, 224)
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return class_labels[predicted_class], confidence

    # ✅ Extract text from image
    def extract_text(img_path):
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return text.strip()

    # ✅ Generate structured response from Gemma 2B
    def generate_text_api(prompt):
        response = client.text_generation(prompt, max_new_tokens=800)
        return response

    # ✅ Store in MySQL
    def store_prediction(company_name, response):
        try:
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
            cursor = conn.cursor()

            prediction_id = str(uuid.uuid4())
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            query = "INSERT INTO predictions (id, company_name, response, date) VALUES (%s, %s, %s, %s)"
            values = (prediction_id, company_name, response, date)
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"❌ MySQL Error: {err}")
            return False

    # ✅ Run Predictions
    with st.spinner("🔍 Processing Image..."):
        company_name, confidence = predict_logo(img_path)
        extracted_text = extract_text(img_path)

        st.subheader("🖼️ Logo Classification")
        st.success(f"Predicted Company: **{company_name}** (Confidence: {confidence:.2f})")

        if company_name in ["Original Ferrari logo", "Original Ford logo"]:
            st.subheader("📜 Extracted Invoice Text")
            st.text_area("Raw Extracted Text:", extracted_text, height=150)

            prompt = f"Extract Invoice Price, Company Name, Order Code, Vehicle configuration values from:\n{extracted_text}"
            response = generate_text_api(prompt)

            st.subheader("🤖 AI-Processed Extraction")
            st.text(response)

            if st.button("💾 Save to Database"):
                if store_prediction(company_name, response):
                    st.success("✅ Data stored successfully in MySQL!")
                else:
                    st.error("⚠️ Failed to store data.")

# 📋 View Stored Predictions
st.sidebar.header("📊 View Predictions")
if st.sidebar.button("🔍 Fetch Stored Data"):
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()
        conn.close()

        st.sidebar.subheader("📜 Past Predictions")
        st.sidebar.write("Showing last 5 entries:")
        for row in rows[-5:]:  # Show last 5 records
            st.sidebar.text(f"{row[1]} - {row[3]}")
    except mysql.connector.Error as err:
        st.sidebar.error(f"❌ MySQL Error: {err}")
