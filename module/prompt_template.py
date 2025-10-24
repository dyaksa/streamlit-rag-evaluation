from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from module.llm_agent import mistral_llm, gemini_llm
from pydantic import BaseModel, Field
import json


class CandidateInfo(BaseModel):
    skills: list[str] = Field(..., description="List of candidate's skills")
    experience: list[str] = Field(
        ..., description="List of candidate's work experience"
    )
    projects: list[str] = Field(..., description="List of candidate's projects")


def extracted_resume(resume_text: str) -> str:
    prompt = """
    Resume:
    {resume_text}
    """

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="""
            You are an expert in talent recruitment. 
            Analyze the following resume and extract key information including skills, experience and projects.
            """,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        ),
    ]

    chat = ChatPromptTemplate(message_templates=messages)
    llm = gemini_llm(temperature=0.0)
    sllm = llm.as_structured_llm(CandidateInfo)
    output = sllm.complete(prompt=chat.format(resume_text=resume_text))

    print("extracted_resume output : ", output)

    raw = output.raw
    if not isinstance(raw, CandidateInfo):
        raise ValueError("Failed to parse candidate information from LLM response.")

    if raw.skills is None:
        raw.skills = []

    if raw.experience is None:
        raw.experience = []

    if raw.projects is None:
        raw.projects = []

    skills = ", ".join(skill for skill in raw.skills)
    experience = "\n".join(exp for exp in raw.experience)
    projects = "\n".join(proj for proj in raw.projects)
    text = f"Skills: {skills}\n\n"
    text += f"Experience:\n{experience}\n\n"
    text += f"Projects:\n{projects}\n"

    return text


class CompareResult(BaseModel):
    cv_match_score: float = Field(
        ...,
        description="Score indicating how well the CV matches the job context range 0-1",
    )
    cv_feedback: str = Field(
        ...,
        description="summarized feedback on the CV in relation to the job context",
    )
    project_score: float = Field(
        ..., description="Score for projects criteria range 0-10"
    )
    project_feedback: str = Field(..., description="Feedback on projects criteria")


def compare_cv_from_job_description(
    cv_text: str, rubric: dict, job_context: str
) -> CompareResult:
    prompt = f"""
    Candidate Info:
    {cv_text}

    Rubric Criteria:
    {json.dumps(rubric, indent=2)}

    Job Context:
    {job_context}

    Return strict JSON:
    """

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="""
            You are an expert in talent recruitment. 
            Compare the candidate's resume info with the job context and provide an analysis of how well the candidate meets the job requirements based on the rubric criteria provided.
            If the rubric criteria and candidate information values are out of range, then just calculate the value based on the facts provided in the data. 
            """,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        ),
    ]

    llm = gemini_llm(temperature=0.0)
    sllm = llm.as_structured_llm(CompareResult)
    output = sllm.chat(messages)

    if not isinstance(output.raw, CompareResult):
        raise ValueError("Failed to parse comparison result from LLM response.")

    return output.raw


class Rubrics(BaseModel):
    skills: float = Field(..., description="Weight for skills criteria range 0-1")
    experiences: float = Field(
        ..., description="Weight for experiences criteria range 0-1"
    )
    projects: float = Field(..., description="Weight for projects criteria range 0-1")


def extract_rubrics_with_llm(job_description: str) -> Rubrics:
    prompt = f"""
    Job Description:
    {job_description}

    Goal: output rubric weights for categories {"skills","experience","projects"}.
    Scoring rules (heuristics):
    - skills_raw = sum over skills of (1.0 * frequency) * (1.5 if required) * (0.7 if preferred)
    - experience_raw = (1 + min(years_required,10)/10 + 0.3 if seniority present) * (1.2 if explicit_must)
    - projects_raw = sum over projects of (1.0 per item) * (1.2 if has_action_verb) * (1.2 if domain_keywords) * (1.2 if is_responsibility_section)

    Then normalize: weight = raw / (skills_raw + experience_raw + projects_raw).

    Constraints:
    - Each in [0,1], rounded to 1 decimal.
    - Sum must be 1.0 after rounding (renormalize if needed).
    - No extra keys, no comments.

    Return strict JSON:
    """

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="""
            You are given extracted signals from a job description:
            """,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        ),
    ]

    llm = gemini_llm(temperature=0.0)
    sllm = llm.as_structured_llm(Rubrics)
    output = sllm.chat(messages)

    raw = output.raw
    if not isinstance(raw, Rubrics):
        raise ValueError("Failed to parse rubric criteria from LLM response.")

    return raw
