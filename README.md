# JOB AI

This project is a Resume-Job Description Matching System built using Streamlit. It leverages OCR (Optical Character Recognition) to extract text from resumes and job descriptions, processes the text using NLP (Natural Language Processing), and matches them using a trained machine learning model. The system helps recruiters or hiring managers quickly evaluate how well a resume fits a job description.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Acknowledgements](#acknowledgements)

---

## Introduction
The Resume-Job Description Matching System is designed to automate the process of evaluating resumes against job descriptions. It uses OCR to extract text from PDFs or images, preprocesses the text using NLP techniques, and applies a machine learning model to predict the fit between the resume and the job description. The system is built with Streamlit, making it easy to use and deploy.

---

## Features
- **OCR for Text Extraction**: Extracts text from PDFs and images using EasyOCR and pdfminer.
- **NLP Preprocessing**: Cleans and preprocesses text using spaCy and NLTK.
- **Skill Extraction**: Identifies skills from resumes and job descriptions using a custom spaCy entity ruler.
- **Machine Learning Model**: Predicts the fit between a resume and a job description using a trained Decision Tree model.
- **Streamlit UI**: Provides an intuitive and interactive web interface for users to upload resumes and job descriptions.

---

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/adnanbhat7/Job_AI.git
   cd Job_AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download additional resources:
   - Download the spaCy English model:
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - Download NLTK data:
     ```python
     import nltk
     nltk.download('words')
     ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Open the Streamlit app in your browser.
2. Upload a resume (PDF or image) and a job description (PDF or image).
3. The app will extract text, preprocess it, and display the matching results.
4. The system will classify the resume into one of the following categories:
   - **GOOD FIT**: The resume is a strong match for the job description.
   - **POTENTIAL**: The resume has some potential but may lack certain skills.
   - **NO FIT**: The resume does not match the job description.

---

## How It Works

1. **Text Extraction**:
   - For PDFs, the app uses pdfminer to extract text.
   - If text extraction fails, it falls back to EasyOCR for OCR-based text extraction from images.

2. **Text Preprocessing**:
   - The text is cleaned and preprocessed using spaCy and NLTK.
   - Stopwords, punctuation, and unnecessary symbols are removed.

3. **Skill Extraction**:
   - A custom spaCy entity ruler is used to identify skills in the text.

4. **Feature Extraction**:
   - Features such as Jaccard similarity for skills, adjectives, and adverbs are calculated.

5. **Prediction**:
   - A trained Decision Tree model predicts the fit between the resume and job description.

6. **Results**:
   - The app displays the predicted fit, common skills, and missing skills.
---

## Acknowledgements
- **Streamlit** - For the interactive web app framework.
- **EasyOCR** - For OCR functionality.
- **spaCy** - For NLP preprocessing and skill extraction.
- **pdfminer** - For PDF text extraction.
- **scikit-learn** - For the machine learning model.
