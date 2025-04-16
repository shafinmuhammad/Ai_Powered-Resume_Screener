import os
import re
import pickle
import PyPDF2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import words
from nltk.metrics.distance import edit_distance

# Download NLTK words corpus for typo correction (run once if needed)
import nltk
nltk.download('words')
word_list = set(words.words())

# Define the 24 job roles
job_roles = [
    "accountant", "advocate", "agriculture", "apparel", "arts", "automobile", "aviation",
    "banking", "bpo", "business-development", "chef", "construction", "consultant",
    "designer", "digital-media", "engineering", "finance", "fitness", "healthcare",
    "hr", "information-technology", "public-relations", "sales", "teacher"
]

# Function to correct OCR typos
def correct_typo(word):
    if word.lower() in word_list:
        return word.lower()
    candidates = [w for w in word_list if len(w) >= len(word) - 2 and len(w) <= len(word) + 2]
    if candidates:
        closest = min(candidates, key=lambda w: edit_distance(word.lower(), w))
        if edit_distance(word.lower(), closest) <= 2:
            return closest
    return word.lower()

# Extract and preprocess text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
            # Clean and correct typos
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'[^\w\s-]', '', text)  # Keep letters, numbers, hyphens
            words = text.split()
            corrected_words = [correct_typo(word) for word in words]
            return " ".join(corrected_words) if corrected_words else None
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Load resumes and labels
resume_folder = r"C:\Users\PC\OneDrive\Desktop\shafin\Project_ind\data"  # Update this path
data = []
labels = []

print("Loading resumes for 24 job roles...")
for job_role in os.listdir(resume_folder):
    job_path = os.path.join(resume_folder, job_role)
    if os.path.isdir(job_path) and job_role.lower() in [jr.lower() for jr in job_roles]:
        print(f"Processing folder: {job_role}")
        for pdf_file in os.listdir(job_path):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(job_path, pdf_file)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    data.append(text)
                    labels.append(job_role)
                else:
                    print(f"Skipping {pdf_file} due to extraction failure")

if not data:
    print("No data loaded. Check folder structure or PDF files.")
    exit()

# Convert to DataFrame
df = pd.DataFrame({"text": data, "label": labels})
print(f"Loaded {len(df)} resumes across {df['label'].nunique()} job roles.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Split into {len(X_train)} training and {len(X_test)} testing samples.")

# Train the model
print("Training model for 24 job roles...")
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save the model
with open("resume_classifier_24_roles.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to resume_classifier_24_roles.pkl")