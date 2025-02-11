import streamlit as st
import os
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import easyocr
import cv2
import spacy
import pandas as pd
import joblib
from tempfile import NamedTemporaryFile
import nltk
nltk.download('words')
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.preprocessing import LabelEncoder

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load Spacy model
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.from_disk("DATA/skills.jsonl")

# Load trained model
@st.cache_resource
def load_model():
    with open("decision_tree.pkl", "rb") as f:
        return joblib.load(f)

model = load_model()

# Initialize NLP pipeline
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    skill_path = "DATA/skills.jsonl"
    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk(skill_path)
    return nlp

nlp = load_spacy_model()

# Load the LabelEncoder to decode predictions
labels = ['GOOD FIT', 'NO FIT', 'POTENTIAL']  # Ensure this matches the encoding order
label_encoder = LabelEncoder()
label_encoder.fit(labels)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfminer or OCR if needed"""
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            raise Exception("No text found")
        return text
    except:
        images = convert_from_path(pdf_path, dpi=300)
        full_text = []
        for page in images:
            img_path = "temp_page.jpg"
            page.save(img_path, 'JPEG')
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            denoised = cv2.medianBlur(binary, 3)
            result = reader.readtext(denoised)
            page_text = " ".join([detection[1] for detection in result])
            full_text.append(page_text)
            os.remove(img_path)
        return " ".join(full_text)

def preprocessing(text):
    """Clean and preprocess text"""
    text = re.sub(r"[.,\-|•]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower().strip() 
        for token in doc 
        if token.text not in STOP_WORDS
        and token.pos_ not in ['PUNCT', 'SYM', 'SPACE']
    ])

def get_features(resume_text, jd_text):
    """Extract features from resume and job description texts"""
    # Preprocess texts
    pre_resume = preprocessing(resume_text)
    pre_jd = preprocessing(jd_text)
    
    # Extract skills
    resume_skills = set(ent.text for ent in nlp(pre_resume).ents if ent.label_ == 'SKILL')
    jd_skills = set(ent.text for ent in nlp(pre_jd).ents if ent.label_ == 'SKILL')
    
    # Extract adjectives and adverbs
    resume_adj = set(token.lemma_.lower() for token in nlp(resume_text) if token.pos_ == "ADJ")
    jd_adj = set(token.lemma_.lower() for token in nlp(jd_text) if token.pos_ == "ADJ")
    
    resume_adv = set(token.lemma_.lower() for token in nlp(resume_text) if token.pos_ == "ADV")
    jd_adv = set(token.lemma_.lower() for token in nlp(jd_text) if token.pos_ == "ADV")
    
    # Calculate Jaccard similarities
    def jaccard(a, b):
        if not a and not b: return 0
        return len(a & b) / len(a | b)
    
    common_skills = resume_skills & jd_skills
    missing_skills_in_resume = jd_skills - resume_skills
    
    return {
        'jaccard_skills': jaccard(resume_skills, jd_skills),
        'jaccard_adj': jaccard(resume_adj, jd_adj),
        'jaccard_adv': jaccard(resume_adv, jd_adv),
        'common_skills': common_skills,
        'missing_skills_in_resume': missing_skills_in_resume
    }

# Streamlit UI
st.title("Resume-Job Description Matching System")

resume_file = st.file_uploader("Upload Resume PDF", type="pdf")
jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")

if st.button("Analyze"):
    if resume_file and jd_file:
        with st.spinner("Processing documents..."):
            # Save uploaded files
            with NamedTemporaryFile(delete=False) as tmp_resume, NamedTemporaryFile(delete=False) as tmp_jd:
                tmp_resume.write(resume_file.read())
                tmp_jd.write(jd_file.read())
                
                # Extract texts
                resume_text = extract_text_from_pdf(tmp_resume.name)
                jd_text = extract_text_from_pdf(tmp_jd.name)
                
            # Remove temporary files
            os.unlink(tmp_resume.name)
            os.unlink(tmp_jd.name)
            
            # Extract features
            features = get_features(resume_text, jd_text)
            feature_df = pd.DataFrame([features])
            
            # Make prediction - Only pass the relevant features (numeric ones) to the model
            prediction = model.predict(feature_df[['jaccard_skills', 'jaccard_adj', 'jaccard_adv']])[0]
            probability = model.predict_proba(feature_df[['jaccard_skills', 'jaccard_adj', 'jaccard_adv']])[0]
            
            # Map prediction back to original label
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            
            # Display results
            st.subheader("Results")
            st.metric("Prediction", prediction_label, f"{probability[prediction]:.0%} confidence")
            
            st.subheader("Similarity Scores")
            col1, col2, col3 = st.columns(3)
            col1.metric("Skills Similarity", f"{features['jaccard_skills']:.0%}")
            col2.metric("Adjectives Similarity", f"{features['jaccard_adj']:.0%}")
            col3.metric("Adverbs Similarity", f"{features['jaccard_adv']:.0%}")
            
            # Display common and missing skills
            st.subheader("Skills Comparison")
            st.write("Common Skills:", ", ".join(features['common_skills']))
            st.write("Missing Skills in Resume (Needed by JD):", ", ".join(features['missing_skills_in_resume']))
            
    else:
        st.error("Please upload both a resume and job description PDF")
