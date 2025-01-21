import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import Field
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from litellm import Router
from extract_thinker import Extractor, Contract, DocumentLoaderPyPdf, LLM
from docx import Document
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ✅ Define the Resume Contract
class ResumeContract(Contract):
    name: str = Field("First and Last Name")
    email: str = Field("Email address")
    phone: Optional[str] = Field("Phone number")
    university: str = Field("University name")
    year_of_study: Optional[int] = Field("Year of study")
    course: str = Field("Course name")
    discipline: str = Field("Discipline")
    cgpa: Optional[float] = Field("CGPA or Percentage")
    key_skills: List[str] = Field("List of key skills")
    gen_ai_experience: int = Field(
        "Gen AI Experience Score (1-3). 1 - Exposed, 2 - Hands-on, 3 - Advanced (Agentic RAG, Evals, etc.)"
    )
    ai_ml_experience: int = Field(
        "AI/ML Experience Score (1-3). 1 - Exposed, 2 - Hands-on, 3 - Advanced (Agentic RAG, Evals, etc.)"
    )
    supporting_info: str = Field("Additional insights such as projects,publications,awards.etc. Ensure adaptability to resume variations.")

# ✅ Configure LiteLLM Router
def config_router():
    rpm = 100  
    model_list = [
        {"model_name": "Groq-Llama3-8B", "litellm_params": {"model": "groq/llama3-8b-8192", "api_key": os.getenv("GROQ_API_KEY"), "rpm": rpm}},
        {"model_name": "Groq-Mixtral-8x7B", "litellm_params": {"model": "groq/mixtral-8x7b-32768", "api_key": os.getenv("GROQ_API_KEY"), "rpm": rpm}},
    ]
    router = Router(model_list=model_list, default_fallbacks=["Groq-Llama3-8B"], set_verbose=True)
    return router

# ✅ Google Drive Integration to Fetch Resumes (PDF & DOCX)
def download_resumes_from_drive(folder_id):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = 'service_account.json'

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    os.makedirs("resumes", exist_ok=True)
    resume_paths = []

    for file in files:
        request = service.files().get_media(fileId=file['id'])
        file_path = os.path.join("resumes", file['name'])
        with open(file_path, "wb") as f:
            f.write(request.execute())
        resume_paths.append(file_path)
    
    return resume_paths

# ✅ Process Resumes & Extract Information
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_resume(resume_path, extractor):
    if resume_path.endswith(".pdf"):
        result = extractor.extract(resume_path, ResumeContract)
    elif resume_path.endswith(".docx"):
        text = extract_text_from_docx(resume_path)
        result = extractor.extract(text, ResumeContract)
    return json.loads(result.json())

def process_resumes_in_batches(resume_paths):
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())

    router = config_router()
    llm = LLM("Groq-Llama3-8B")
    llm.load_router(router)
    extractor.load_llm(llm)

    extracted_data = []
    batch_size = 10  # Number of resumes to process concurrently

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for result in tqdm(executor.map(lambda path: process_resume(path, extractor), resume_paths), total=len(resume_paths)):
            extracted_data.append(result)

    df = pd.DataFrame(extracted_data)
    df.to_excel("Extracted_Resumes.xlsx", index=False)
    return df

# ✅ Streamlit UI
st.title("📄 Resume Analyzer")
st.sidebar.header("Upload Resumes via Google Drive")

folder_id = st.sidebar.text_input("Google Drive Folder ID", "")
if st.sidebar.button("Analyze Resumes"):
    if folder_id:
        resume_paths = download_resumes_from_drive(folder_id)
        if resume_paths:
            df = process_resumes_in_batches(resume_paths)
            st.success(f"✅ {len(df)} resumes processed successfully!")
            st.dataframe(df)

            # Visualization
            st.subheader("📊 Data Visualization")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Experience Scores
            sns.countplot(data=df, x="gen_ai_experience", ax=axes[0], palette="coolwarm")
            axes[0].set_title("Gen AI Experience Score Distribution")
            axes[0].set_xlabel("Experience Level (1-3)")
            axes[0].set_ylabel("Count")

            sns.countplot(data=df, x="ai_ml_experience", ax=axes[1], palette="coolwarm")
            axes[1].set_title("AI/ML Experience Score Distribution")
            axes[1].set_xlabel("Experience Level (1-3)")
            axes[1].set_ylabel("Count")

            st.pyplot(fig)

            st.download_button("📥 Download Extracted Data",
                        data=open("Extracted_Resumes.xlsx", "rb").read(),
                        file_name="Extracted_Resumes.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"    
                )

        else:
            st.error("❌ No resumes found in the given folder.")
    else:
        st.warning("⚠ Please enter a Google Drive Folder ID.")
