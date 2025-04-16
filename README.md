🧠 AI-Powered Resume Screener
The AI-Powered Resume Screener is a Streamlit-based web application designed to automate resume analysis for recruitment. 
It evaluates PDF resumes against 24 job roles and user-selected skills, providing job fit scores and detailed candidate insights.
The app features two interfaces: one for analyzing a single resume and another for comparing up to 10 resumes in a table format. 
This tool streamlines hiring by reducing manual effort and enabling customized skill matching.

🚀 Project Overview
This project is designed to automate the resume screening process by:

Predicting job roles from resume content

Extracting key details like skills, education, experience, and languages

Comparing resumes against selected job roles and skill requirements

Offering both single resume analysis and bulk resume comparison

Built using:

🐍 Python

📚 Machine Learning (TF-IDF + Naive Bayes)

📄 PDF Parsing with PyPDF2

🔍 NLP with spaCy

🌐 Streamlit for Web App Deployment


Features



Single Resume Analysis:


Upload one PDF, select job role and skills.

Outputs: Education, experience, skills, languages; job fit %; matching features; skill match %.



Multiple Resume Comparison:


Upload 1–10 PDFs, select job role and skills.

Outputs: Table ranking resumes by job fit, with details and skill matches.

Custom Skill Comparison: Match resumes against chosen skills.



Installation

Prerequisites

Python 3.8+
pip

Dependencies

streamlit
PyPDF2
spacy
scikit-learn
pandas


Setup

Install dependencies: pip install streamlit PyPDF2 spacy scikit-learn pandas

Download spaCy model: python -m spacy download en_core_web_sm

Place resume_classifier.pkl in the project root.

Save app code as app.py.


Technical Details


Frontend: Streamlit (two tabs).

Backend:
PyPDF2: PDF extraction.
spaCy: NLP parsing.
scikit-learn: Naive Bayes model.
pandas: Table output.
Job Roles: 24 (e.g., healthcare, engineering).
Skill Matching: User-selected skills comparison.


Notes

Requires resume_classifier.pkl.

PDFs must have readable text.


