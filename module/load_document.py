from llama_index.readers.file import PDFReader
from pathlib import Path
from typing import List
import re


def _clean_resume_text(text: str) -> str:
    # 1) collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # 2) spacing around pipes & punctuation
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\(\s*", " (", text)
    text = re.sub(r"\s*\)\s*", ") ", text)
    # 3) bullets → newline
    text = text.replace("•", "\n• ")
    # 4) normalize date dashes
    text = re.sub(r"\s*[–—-]\s*", " – ", text)
    # 5) section headings
    for h in [
        "WORKING EXPERIENCE",
        "EDUCATION",
        "CERTIFICATIONS",
        "LANGUAGES AND SKILLS",
    ]:
        text = re.sub(rf"\s*{h}\s*", f"\n\n{h}\n", text, flags=re.I)
    # 6) job headers (PT …) + (non-PT …)
    month = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    text = re.sub(
        rf"\s+(PT [A-Z][A-Za-z0-9 &.-]+,\s*[A-Za-z ]+\s+{month}\s+\d{{4}}\s+–\s+(?:Present|{month}\s+\d{{4}}))",
        r"\n\n\1",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"\s+([A-Z][A-Za-z0-9 '&.-]+(?: [A-Z][A-Za-z0-9 '&.-]+)*\s*,\s*[A-Za-z ]+\s+{month}\s+\d{{4}}\s+–\s+(?:Present|{month}\s+\d{{4}}))",
        r"\n\n\1",
        text,
        flags=re.I,
    )
    # 7) remove “CV updated …”
    text = re.sub(
        r"CV updated on [A-Za-z]{3,9}\s+\d{1,2},\s*\d{4},?", "", text, flags=re.I
    )
    # 8) compact blanks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 9) phone +62 spacing
    text = re.sub(r"\+62(?:\s+\d)+", lambda m: " ".join(m.group(0).split()), text)
    # 10) keep first line compact as header
    lines = text.splitlines()
    if lines:
        header_tokens = []
        while lines and ("WORKING EXPERIENCE" not in lines[0].upper()):
            if len(lines[0]) > 140 and lines[0].count("|") == 0:
                break
            header_tokens.append(lines.pop(0))
            if "|" in header_tokens[-1]:
                break
        header = re.sub(r"\s{2,}", " ", " ".join(header_tokens)).strip()
        text = "\n".join([header] + lines)
    # 11) final punctuation spacing cleanup
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+;", ";", text)
    text = re.sub(r"\n•\s*", "\n• ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _preprocessed_text(texts: List[str]) -> str:
    if not texts:
        return ""

    text = ""
    for t in texts:
        t = re.sub(r"[^0-9A-Za-z ]", "", t)
        t = t.lower()
        t = re.sub(r"\s+", " ", t).strip()
        text += t + " "

    return text.strip()


def load_pdf_document(file_path: str) -> str:
    reader = PDFReader()
    document = reader.load_data(file=Path(file_path), extra_info={"source": file_path})
    text = ""
    for doc in document:
        text += _clean_resume_text(doc.text.replace("\n", " ")) + "\n"
    return text.strip()


def load_job_description(job_description: str):
    cleaned = _preprocessed_text([job_description])
    return cleaned
