# FORD-FERRARI-Invoice-Analyzer
The FORD & FERRARI Invoice Analyzer is a "Streamlit-based app" for real-time invoice classification and extraction. It detects Ford/Ferrari logos using a "CNN model", extracts text via "Tesseract OCR", processes details with "Gemma 2B (Hugging Face API)", and stores results in a "MySQL database". 


Key Features

1) CNN-based Invoice Logo Classification:
Uses a pre-trained Convolutional Neural Network (CNN) model to classify invoice images based on their logos (Ford or Ferrari).
Identifies logos with confidence scores to ensure reliable classification.

2) OCR-based Text Extraction:
Utilizes Tesseract OCR to extract invoice details such as price, order code, vehicle configuration, etc.
Converts printed invoice text into a machine-readable format.

3) AI-Powered Invoice Processing:
Uses Hugging Face's Gemma 2B model via Hugging Face API to extract key structured details from invoice text.
Generates structured information like invoice price, order code, and vehicle details based on extracted text.

4) Database Storage (MySQL):
Stores predictions, extracted invoice text, and AI-generated structured responses in a MySQL database.
Allows users to fetch and view stored invoice data for future reference.

5) User-Friendly Web Interface (Streamlit):
Enables users to upload images, view predictions, and save extracted data seamlessly.
Provides a sidebar to fetch and display previously stored invoices.
