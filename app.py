import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS to change the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\\n"
    return text.strip() if text else "No readable text found."

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

st.title("AI Resume Screening & Ranking System")
st.write("Upload resumes and enter the job description to rank the resumes based on their relevance to the job description.")

job_description = st.text_area("Enter the job description")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    with st.spinner('Processing resumes...'):
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)

        ranked_resumes = sorted(zip(uploaded_files, resumes, scores), key=lambda x: x[2], reverse=True)
    
    st.subheader("Ranked Resumes")
    for i, (file, resume_text, score) in enumerate(ranked_resumes, start=1):
        st.write(f"### {i}. {file.name} - Score: {score:.2f}")
        with st.expander("Show Resume Text"):
            st.write(resume_text)
    
    st.success('Resumes have been ranked successfully!')
else:
    st.info("Please upload resumes and enter the job description to start the ranking process.")
