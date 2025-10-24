from huggingface_hub import upload_file
from module.load_document import load_pdf_document, load_job_description
from module.splitter import split_text_into_chunks
from module.collection import (
    retrieve,
    create_query_fusion_retriever,
)
from module.embedding_agent import get_embedding_mistral
from module.llm_agent import mistral_llm, gemini_llm
from module.prompt_template import (
    extracted_resume,
    compare_cv_from_job_description,
    extract_rubrics_with_llm,
)
import streamlit as st
import warnings
import tempfile
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


def save_upload_to_disk(uploaded_file, suffix=".pdf"):
    """Persist uploaded file with a timestamped filename.

    Pattern: originalName_without_ext_YYYY-MM-DD_HHMMSS.ext
    If no original name provided, uses 'upload'.
    """
    key = f"uploaded_id_{uuid.uuid4()}"
    if key in st.session_state and Path(st.session_state[key]).exists():
        return st.session_state[key]

    if "session_tmpdir" not in st.session_state:
        st.session_state.session_tmpdir = tempfile.mkdtemp(prefix="st_resume_eval_")

    original_name = uploaded_file.name or "upload"
    base, ext = os.path.splitext(original_name)
    if not ext:
        ext = suffix

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_base = base.replace(" ", "_").replace("/", "_")
    new_filename = f"{safe_base}_{timestamp}{ext}".lower()

    out_path = Path(st.session_state.session_tmpdir) / new_filename

    with open(out_path, "wb") as f:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, f)

    st.session_state[key] = str(out_path)
    return st.session_state[key]


def streamlit_interface():
    try:
        st.title("Resume Evaluation with LLM 🧠")

        uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])
        job_description = st.text_area("Enter Job Description", height=200)
        button = st.button("Evaluate Resume")

        if button and uploaded_file is not None and job_description.strip() != "":
            # Show a spinner while heavy processing runs
            with st.spinner("Evaluating resume... This can take a few seconds."):
                # Optional status area for granular updates
                status = st.empty()

                status.markdown("✅ Upload received. Parsing PDF...")
                temp_file_path = save_upload_to_disk(uploaded_file, suffix=".pdf")
                document = load_pdf_document(temp_file_path)
                candidate_info = extracted_resume(document)

                status.markdown("📋 Preprocessing job description...")
                preprocessed_job_description = load_job_description(job_description)

                status.markdown("🧩 Splitting job description into chunks...")
                chunks = split_text_into_chunks(
                    preprocessed_job_description, chunk_size=100, chunk_overlap=10
                )

                status.markdown("🧪 Extracting rubrics with LLM...")
                rubrics = extract_rubrics_with_llm(preprocessed_job_description)
                st.subheader("Extracted Rubrics Job Description")
                st.json(rubrics.dict())

                status.markdown("🧠 Initializing LLM & embeddings...")
                llm = gemini_llm(temperature=0.0)

                embedding = get_embedding_mistral()
                index = create_query_fusion_retriever(
                    chunks, embedding=embedding, llm=llm, top_k=3
                )

                status.markdown("🔍 Retrieving relevant job context...")
                query_resume = "Which part can I use for CV scoring?"
                response = retrieve(index, query_resume, top_k=3)

                job_context = ".\n".join([res.text for res in response])
                st.subheader("Job Context")
                st.text(job_context)

                status.markdown("⚖️ Comparing resume against job description...")
                comparison_result = compare_cv_from_job_description(
                    candidate_info,
                    {
                        "skills": rubrics.skills,
                        "experiences": rubrics.experiences,
                        "projects": rubrics.projects,
                    },
                    job_context,
                )

                status.markdown("✅ Evaluation complete.")
                st.subheader("Comparison Result")
                st.json(comparison_result.dict())
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    streamlit_interface()
