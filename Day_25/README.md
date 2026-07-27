# Document OCR Web Application

## Overview

This project is a Document OCR Web Application developed using Python, EasyOCR, OpenCV, and Gradio.

The application allows users to upload a document image, preprocess it, extract readable text using OCR, 
display the extracted text, and download it as a text file.

## Features

- Upload document images
- Image preprocessing
  - Grayscale conversion
  - Gaussian Blur
  - Otsu Thresholding
- Text extraction using EasyOCR
- Display the original uploaded image
- Display extracted text
- Download extracted text as a '.txt' file
- Supports multiple document types:
  - Documents
  - Receipts
  - Invoices
  - Forms

## Technologies Used

- Python
- EasyOCR
- OpenCV
- Gradio
- NumPy

## Project Structure

Document_OCR/
│
├── app.py
├── README.md
├── requirements.txt
├── images/
├── output/
└── .gitignore


## How to Use

1. Upload a document image.
2. The application preprocesses the image.
3. EasyOCR extracts readable text.
4. The original image and extracted text are displayed.
5. Download the extracted text using the provided button.


## Supported Documents

- Printed documents
- Receipts
- Invoices
- Forms


## Future Improvements

- Handwritten text recognition
- Multiple language support
- PDF document OCR
- Improved preprocessing techniques