import sqlite3
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from datetime import datetime
import random
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

# Initialize the database
DB_FILE = "resume_enhancer.db"

def create_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            alignment_score REAL NOT NULL,
            semantic_similarity REAL NOT NULL,
            tfidf_similarity REAL NOT NULL,
            keyword_match_score REAL NOT NULL,
            timestamp DATETIME NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def upgrade_database():
    """Ensure the `analytics` table has all required columns."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE analytics ADD COLUMN semantic_similarity REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE analytics ADD COLUMN tfidf_similarity REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE analytics ADD COLUMN keyword_match_score REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# Create and upgrade the database
create_database()
upgrade_database()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = "".join([page.extract_text() for page in reader.pages])
    return text

# Enhanced Function to Find Gaps
def find_gaps(resume_text, job_text):
    # TF-IDF Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Semantic Similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a sentence transformer model
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_text)
    semantic_similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

    # Calculate a weighted average
    alignment_score = 0.7 * semantic_similarity + 0.3 * tfidf_similarity
    return alignment_score, semantic_similarity, tfidf_similarity

# Keyword Matching Function
def keyword_matching(resume_text, job_text):
    resume_keywords = set(resume_text.split())
    job_keywords = set(job_text.split())
    common_keywords = resume_keywords.intersection(job_keywords)
    match_score = len(common_keywords) / len(job_keywords) * 100  # Percentage match
    return list(common_keywords), match_score

# Function to log analytics into the database
def log_analytics(user_id, role, alignment_score, semantic_similarity, tfidf_similarity, keyword_match_score):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analytics (user_id, role, alignment_score, semantic_similarity, tfidf_similarity, keyword_match_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, role, alignment_score, semantic_similarity, tfidf_similarity, keyword_match_score, datetime.now()))
    conn.commit()
    conn.close()

# Function to retrieve and display analytics
def display_analytics():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM analytics", conn)
    conn.close()
    if not df.empty:
        st.write("### Alignment Analytics")
        st.dataframe(df)
    else:
        st.write("No analytics data available yet.")

# Dynamic Certification Suggestions
CERTIFICATION_POOL = [
    "Coursera: Python for Everybody",
    "Udemy: Complete Python Bootcamp",
    "LinkedIn Learning: Advanced Excel",
    "Coursera: Data Science Specialization",
    "Google: Data Analytics Professional Certificate",
    "AWS: Cloud Practitioner Essentials",
    "Udemy: Machine Learning A-Z",
    "Coursera: Communication Foundations"
]

def dynamic_certifications():
    num_suggestions = random.randint(2, 5)
    return random.sample(CERTIFICATION_POOL, num_suggestions)

# Function to enhance resume section
def enhance_resume_section(section, missing_skills, job_title):
    return f"{section}\n\nSkills added for {job_title}: {', '.join(missing_skills)}."

# Initialize the ChatGroq model
chat_model = ChatGroq(api_key=os.getenv("CHATGROQ_API_KEY"))

# Function to generate enhanced suggestions using ChatGroq
def generate_enhanced_suggestion(text, role):
    prompt = f"""
    Enhance the following section of a resume for the job role "{role}":
    {text}
    Focus on the key skills and experience relevant to "{role}".
    """
    response = chat_model.predict(text=prompt)
    return response if isinstance(response, str) else "Error generating suggestion."

# Use session state for input persistence
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "alignment_score" not in st.session_state:
    st.session_state.alignment_score = 0.0
if "semantic_similarity" not in st.session_state:
    st.session_state.semantic_similarity = 0.0
if "tfidf_similarity" not in st.session_state:
    st.session_state.tfidf_similarity = 0.0
if "keyword_match_score" not in st.session_state:
    st.session_state.keyword_match_score = 0.0

# Streamlit UI
st.title("AI Resume Enhancer")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_description = st.text_area("Enter Job Description", height=200)

if uploaded_resume and job_description:
    resume_text = extract_text_from_pdf(uploaded_resume)
    job_text = job_description

    st.subheader("Extracted Resume Content")
    st.text_area("Resume", value=resume_text, height=200)

    st.subheader("Job Description")
    st.text_area("Job Description", value=job_text, height=200)

    if st.button("Analyze Alignment"):
        alignment_score, semantic_similarity, tfidf_similarity = find_gaps(resume_text, job_text)
        common_keywords, keyword_match_score = keyword_matching(resume_text, job_text)

        st.session_state.alignment_score = alignment_score
        st.session_state.semantic_similarity = semantic_similarity
        st.session_state.tfidf_similarity = tfidf_similarity
        st.session_state.keyword_match_score = keyword_match_score

        st.write(f"### Resume Alignment Score: {alignment_score:.2f}")
        st.write(f"- Semantic Similarity: {semantic_similarity:.2f}")
        st.write(f"- TF-IDF Similarity: {tfidf_similarity:.2f}")
        st.write(f"- Keyword Match Score: {keyword_match_score:.2f}%")

        st.write("### Common Keywords:", ", ".join(common_keywords))

    st.text_input("Enter User ID", key="user_id")
    st.text_input("Enter Desired Role", key="role")

    if st.session_state.user_id and st.session_state.role:
        log_analytics(
            st.session_state.user_id,
            st.session_state.role,
            st.session_state.alignment_score,
            st.session_state.semantic_similarity,
            st.session_state.tfidf_similarity,
            st.session_state.keyword_match_score
        )

    missing_skills = ["Communication", "Team Management"]  # Update dynamically
    st.write("### Missing Skills:", ", ".join(missing_skills))

    certifications = dynamic_certifications()
    st.write("### Recommended Certifications:")
    for cert in certifications:
        st.write(f"- {cert}")

    st.subheader("Rewrite Suggestions")
    for section in ["Experience", "Skills"]:
        enhanced_section = generate_enhanced_suggestion(section, st.session_state.role)
        st.write(f"**Enhanced {section}:**\n{enhanced_section}")

# Function to delete the most recent entry
def delete_recent_entry():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM analytics ORDER BY timestamp DESC LIMIT 1")
    result = cursor.fetchone()

    if result:
        cursor.execute("DELETE FROM analytics WHERE id = ?", (result[0],))
        conn.commit()
        st.success("The most recent entry has been deleted.")
    else:
        st.warning("No entries found to delete.")

    conn.close()

st.subheader("Analytics")
if st.button("Delete Most Recent Entry"):
    delete_recent_entry()

display_analytics()

st.subheader("Multilingual Support")
language = st.selectbox("Select Language", ["English", "Spanish", "French", "German"])
if language != "English":
    st.write(f"Resume enhancement in {language} coming soon!")
