import   streamlit as st
import pickle
import PyPDF2
import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
try:
    with open("resume_classifier.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'resume_classifier.pkl' not found. Please train the model first.")
    st.stop()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define job roles
job_roles = [
    "accountant", "advocate", "agriculture", "apparel", "arts", "automobile", "aviation",
    "banking", "bpo", "business-development", "chef", "construction", "consultant",
    "designer", "digital-media", "engineering", "finance", "fitness", "healthcare",
    "hr", "information-technology", "public-relations", "sales", "teacher"
]

# Define skill keywords for feature selection
skill_keywords = [
    # Accountant
    "accounting", "financial modeling", "reconciliation", "variance analysis", "cash forecasting", "excel", "inventory accounting", "fixed assets",
    # Advocate
    "legal", "advocacy", "litigation", "research", "negotiation",
    # Agriculture
    "agriculture", "crop management", "irrigation", "livestock", "sustainability",
    # Apparel
    "merchandising", "sales", "customer service", "visual merchandising", "inventory",
    # Arts
    "arts", "program development", "event planning", "grant writing", "budgeting",
    # Automobile
    "automotive", "mechanical", "customer service", "repair", "diagnostics",
    # Aviation
    "aviation", "maintenance", "troubleshooting", "faa regulations", "mechanical aptitude",
    # Banking
    "banking", "foreclosure", "financial analysis", "compliance", "customer relations",
    # BPO
    "analysis", "team leadership", "process improvement", "data management", "communication",
    # Business-Development
    "business development", "sales strategy", "market research", "networking", "negotiation",
    # Chef
    "cooking", "menu planning", "culinary arts", "food safety", "kitchen management",
    # Construction
    "construction", "project management", "safety regulations", "scheduling", "engineering",
    # Consultant
    "consulting", "problem solving", "strategy", "client relations", "data analysis",
    # Designer
    "design", "ux", "ui", "html", "css", "prototyping", "user research",
    # Digital-Media
    "digital media", "content creation", "multimedia", "project coordination", "editing",
    # Engineering
    "engineering", "technical skills", "problem solving", "design", "testing",
    # Finance
    "finance", "financial analysis", "budgeting", "forecasting", "excel",
    # Fitness
    "fitness", "training", "exercise science", "customer service", "program design",
    # Healthcare
    "healthcare", "nursing", "patient care", "medical terminology", "emr",
    # HR
    "hr", "staff management", "training", "policy development", "recruitment",
    # Information-Technology
    "information technology", "networking", "security", "java", "sql", "troubleshooting",
    # Public-Relations
    "public relations", "event planning", "communication", "marketing", "media relations",
    # Sales
    "sales", "negotiation", "customer relationship", "marketing", "analytics",
    # Teacher
    "teaching", "curriculum development", "leadership", "assessment", "instruction"
]

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Extract details without name
def extract_details(text):
    doc = nlp(text)
    skills, education, experience, languages = [], [], [], []

    # Education extraction
    education_keywords = ["university", "college", "school", "institute", "degree"]
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(kw in ent.text.lower() for kw in education_keywords):
            education.append(ent.text)

    # Experience extraction
    date_pattern = r"\d{2}/\d{4}\s*(?:to|-)\s*(?:\d{2}/\d{4}|current)|(?:january|february|march|april|may|june|july|august|september|october|november|december)?\s+\d{4}\s*(?:to|-)\s*(?:\d{4}|current)|\d{4}"
    matches = re.findall(date_pattern, text.lower())
    for match in matches:
        experience.append(match)

    # Skills
    for token in doc:
        if token.text.lower() in skill_keywords and token.text.lower() not in skills:
            skills.append(token.text.lower())

    # Languages
    language_keywords = ["english", "spanish", "french", "german", "hindi", "russian", "deutsch"]
    for token in doc:
        if token.text.lower() in language_keywords and token.text.lower() not in languages:
            languages.append(token.text.lower())

    return {"skills": skills, "education": education, "experience": experience, "languages": languages}

# Extract top TF-IDF features
def get_matching_features(cleaned_text, job_role):
    tfidf = model.named_steps['tfidfvectorizer']
    nb = model.named_steps['multinomialnb']
    feature_names = tfidf.get_feature_names_out()
    resume_vector = tfidf.transform([cleaned_text]).toarray()[0]
    job_idx = job_roles.index(job_role)
    class_log_probs = nb.feature_log_prob_[job_idx]
    feature_scores = resume_vector * class_log_probs
    top_indices = feature_scores.argsort()[-5:][::-1]
    return [feature_names[i] for i in top_indices if feature_scores[i] > 0]

# Compare selected features
def compare_selected_features(extracted_skills, selected_features):
    if not selected_features:
        return {"matched": [], "percentage": 0.0}
    matched = [skill for skill in selected_features if skill in extracted_skills]
    percentage = (len(matched) / len(selected_features)) * 100 if selected_features else 0.0
    return {"matched": matched, "percentage": round(percentage, 2)}

# Streamlit app
st.title("AI-Powered Resume Screener")
st.write("Analyze resumes for job fit using two interfaces.")

# Create two tabs
tab1, tab2 = st.tabs(["Single Resume Analysis", "Multiple Resume Comparison"])

# First Interface: Single Resume Analysis
with tab1:
    st.header("Single Resume Analysis")
    st.write("Upload one resume and compare it to a job role and selected skills.")
    selected_job_single = st.selectbox("Select a job role:", job_roles, key="single_job")
    selected_features_single = st.multiselect("Select skills to compare:", skill_keywords, key="single_features")
    uploaded_file = st.file_uploader("Upload a resume (PDF)", type="pdf", key="single_upload")

    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        if raw_text:
            cleaned_text = clean_text(raw_text)
            with st.spinner("Analyzing resume..."):
                prediction = model.predict([cleaned_text])[0]
                confidence = model.predict_proba([cleaned_text])[0][job_roles.index(selected_job_single)] * 100
                details = extract_details(raw_text)
                matching_features = get_matching_features(cleaned_text, selected_job_single)
                feature_comparison = compare_selected_features(details['skills'], selected_features_single)

            st.subheader("Analysis Results")
            st.write("**Parsed Details**:")
            st.write(f"- Education: {', '.join(details['education']) or 'None detected'}")
            st.write(f"- Experience: {', '.join(details['experience']) or 'None detected'}")
            st.write(f"- Skills: {', '.join(details['skills']) or 'None detected'}")
            st.write(f"- Languages: {', '.join(details['languages']) or 'None detected'}")
            st.write(f"**Predicted Job Role**: {prediction}")
            st.write(f"**Job Fit for {selected_job_single}**: {confidence:.2f}%")
            st.write(f"**Matching Features (Job Role)**: {', '.join(matching_features) or 'None significant'}")
            st.write("**Selected Skills Comparison**:")
            st.write(f"- Matched Skills: {', '.join(feature_comparison['matched']) or 'None matched'}")
            st.write(f"- Match Percentage: {feature_comparison['percentage']:.2f}%")
        else:
            st.error("Could not extract text from the uploaded resume.")

# Second Interface: Multiple Resume Comparison
with tab2:
    st.header("Multiple Resume Comparison")
    st.write("Upload up to 10 resumes to compare against a job role and selected skills.")
    selected_job_multi = st.selectbox("Select a job role:", job_roles, key="multi_job")
    selected_features_multi = st.multiselect("Select skills to compare:", skill_keywords, key="multi_features")
    uploaded_files = st.file_uploader("Upload resumes (PDF, max 10)", type="pdf", accept_multiple_files=True, key="multi_upload")

    if uploaded_files and len(uploaded_files) <= 10:
        results = []
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            if raw_text:
                cleaned_text = clean_text(raw_text)
                with st.spinner(f"Analyzing {file.name}..."):
                    confidence = model.predict_proba([cleaned_text])[0][job_roles.index(selected_job_multi)] * 100
                    details = extract_details(raw_text)
                    matching_features = get_matching_features(cleaned_text, selected_job_multi)
                    feature_comparison = compare_selected_features(details['skills'], selected_features_multi)
                    results.append({
                        "Filename": file.name,
                        "Job Fit (%)": round(confidence, 2),
                        "Education": ", ".join(details['education']) or "None detected",
                        "Experience": ", ".join(details['experience']) or "None detected",
                        "Skills": ", ".join(details['skills']) or "None detected",
                        "Languages": ", ".join(details['languages']) or "None detected",
                        "Matching Features": ", ".join(matching_features) or "None significant",
                        "Selected Skills Match (%)": feature_comparison['percentage'],
                        "Matched Skills": ", ".join(feature_comparison['matched']) or "None matched"
                    })

        if results:
            # Create DataFrame and sort by Job Fit
            df = pd.DataFrame(results)
            df = df.sort_values(by="Job Fit (%)", ascending=False)
            st.subheader(f"Comparison Results for {selected_job_multi}")
            st.dataframe(
                df,
                column_config={
                    "Filename": st.column_config.TextColumn("Resume"),
                    "Job Fit (%)": st.column_config.NumberColumn("Fit (%)", format="%.2f"),
                    "Education": st.column_config.TextColumn("Education"),
                    "Experience": st.column_config.TextColumn("Experience"),
                    "Skills": st.column_config.TextColumn("Skills"),
                    "Languages": st.column_config.TextColumn("Languages"),
                    "Matching Features": st.column_config.TextColumn("Job Features"),
                    "Selected Skills Match (%)": st.column_config.NumberColumn("Skills Match (%)", format="%.2f"),
                    "Matched Skills": st.column_config.TextColumn("Matched Skills")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.error("No valid text extracted from the uploaded resumes.")
    elif uploaded_files:
        st.error("Please upload 10 or fewer resumes.")

