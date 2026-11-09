import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ---------------------------------------------------------
# AI RESUME SCREENING & JOB MATCHING SYSTEM
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening & Job Matching System",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:linear-gradient(180deg,#0b1220,#0f172a);}
[data-testid="stSidebar"] {background:#111827;}
.hero,.candidate {
    padding:1rem 1.2rem; margin-bottom:.8rem;
    border:1px solid #334155; border-radius:16px;
    background:#172033;
}
.muted {color:#aab7ca;}
</style>
""", unsafe_allow_html=True)

STOPWORDS = set("""
a an and are as at be by for from has he in is it its of on or that the to was
were will with this these those i you your we they them their our us me my his
her she him who whom which what when where why how can could should would may
might must do does did done etc plus also using used use
""".split())

SKILLS = {
    "javascript": ["javascript", "ecmascript", "js"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs", "next"],
    "node.js": ["node", "node.js", "nodejs"],
    "express": ["express", "express.js", "expressjs"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "mongodb": ["mongodb", "mongo"],
    "python": ["python", "py"],
    "docker": ["docker", "container"],
    "aws": ["aws", "amazon web services"],
    "rest api": ["rest", "rest api", "rest apis", "api"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "tailwind": ["tailwind", "tailwindcss"],
    "git": ["git", "github", "gitlab"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "sql": ["sql", "database", "db"],
    "machine learning": ["machine learning", "ml", "ai", "artificial intelligence"],
    "data analysis": ["data analysis", "analytics", "analysis"],
    "communication": ["communication", "communicate", "presentation"],
    "ci/cd": ["ci/cd", "cicd", "github actions", "gitlab ci"],
    "kubernetes": ["kubernetes", "k8s"],
    "redis": ["redis"],
    "graphql": ["graphql"],
}

SAMPLE_JOB = """
Full Stack Developer
We are seeking a Full Stack Developer with experience in React, Next.js,
Node.js, Express, PostgreSQL, REST APIs, Docker, AWS, and clean UI development.
Strong communication, problem-solving, and deployment experience are preferred.
"""

SAMPLE_RESUME = """
Sashank Borah
Full Stack Developer
Email: sashank@example.com
Phone: +91-9000000000
LinkedIn: linkedin.com/in/sashank
GitHub: github.com/sashank

Skills: Python, JavaScript, React, Next.js, Node.js, Express, PostgreSQL,
MongoDB, REST APIs, HTML, CSS, Tailwind, Git, Docker, AWS

Experience:
- Built responsive web applications using React and Next.js
- Developed REST APIs with Node.js and Express
- Designed PostgreSQL database schemas and optimized queries
- Worked with Docker, GitHub Actions, and AWS deployment

Education: B.Tech Computer Science
"""


@dataclass
class Candidate:
    name: str
    score: float
    keyword_match: float
    similarity: float
    skill_match: float
    title_match: float
    length_fit: float
    email: bool
    phone: bool
    linkedin: bool
    github: bool
    resume_keywords: List[Tuple[str, int]]
    job_keywords: List[Tuple[str, int]]
    matched_skills: List[str]
    missing_skills: List[str]
    missing_keywords: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    raw_text: str


def normalize(text):
    text = text.lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+.#\s-]", " ", text)).strip()


def tokens(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}", normalize(text))
    return [word for word in words if word not in STOPWORDS]


def keywords(text, limit=20):
    return Counter(tokens(text)).most_common(limit)


def similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def detect_skills(text):
    text = normalize(text)
    return [
        skill for skill, aliases in SKILLS.items()
        if any(alias in text for alias in aliases)
    ]


def read_resume(upload):
    if upload is None:
        return ""
    data = upload.read()
    try:
        if upload.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        if upload.name.lower().endswith(".docx"):
            import docx
            return "\n".join(p.text for p in docx.Document(BytesIO(data)).paragraphs)
        if upload.name.lower().endswith(".txt"):
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def contact_details(text):
    return (
        bool(re.search(r"[\w.-]+@[\w.-]+\.\w+", text)),
        bool(re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)),
        "linkedin" in normalize(text),
        "github" in normalize(text),
    )


def candidate_name(text):
    for line in [x.strip() for x in text.splitlines() if x.strip()][:6]:
        if 2 <= len(line.split()) <= 4 and not re.search(r"[@\d]", line):
            return line
    return "Candidate"


def job_title(text):
    first_line = next((x.strip() for x in text.splitlines() if x.strip()), "")
    return first_line if len(first_line) < 80 else "Target Role"


def screen_resume(resume, job):
    rt, jt = tokens(resume), tokens(job)
    rset, jset = set(rt), set(jt)
    keyword_match = len(rset & jset) / len(rset | jset) if rset | jset else 0
    sim = similarity(resume, job)

    resume_skills, job_skills = detect_skills(resume), detect_skills(job)
    matched = sorted(set(resume_skills) & set(job_skills))
    missing = [s for s in job_skills if s not in resume_skills]
    skill_match = len(matched) / len(set(job_skills)) if job_skills else 0

    title_words = tokens(job_title(job))
    title_match = (
        len(set(title_words) & rset) / len(set(title_words) | rset)
        if title_words and rset else 0.5
    )
    count = len(rt)
    length_fit = 0.35 if count < 120 else 1.0 if count <= 700 else 0.75 if count <= 1200 else 0.55

    email, phone, linkedin, github = contact_details(resume)
    score = (
        keyword_match * .24 + sim * .22 + skill_match * .26
        + title_match * .08 + length_fit * .10
        + (email * .05) + (phone * .03) + (linkedin * .02)
    )
    score = max(0, min(score, 1))

    strengths, weaknesses = [], []
    if skill_match >= .6:
        strengths.append("Strong match with the required skills.")
    if sim >= .45:
        strengths.append("Resume content is closely related to the job description.")
    if email and phone:
        strengths.append("Email and phone details are available.")
    if linkedin or github:
        strengths.append("A professional profile link is available.")
    if len(matched) >= 5:
        strengths.append("The candidate has a good range of relevant skills.")

    for present, label in [
        (email, "Email address"), (phone, "Phone number"),
        (linkedin, "LinkedIn profile"), (github, "GitHub profile")
    ]:
        if not present:
            weaknesses.append(f"{label} not detected.")
    if missing:
        weaknesses.append("Missing skills: " + ", ".join(missing[:6]) + ".")
    if count < 120:
        weaknesses.append("Resume is short, so the result may be less reliable.")

    missing_words = [w for w, _ in Counter(jt).most_common(35) if w not in rset][:12]
    recommendation = "Shortlist" if score >= .70 else "Review" if score >= .45 else "Reject"

    return Candidate(
        candidate_name(resume), score, keyword_match, sim, skill_match,
        title_match, length_fit, email, phone, linkedin, github,
        keywords(resume), keywords(job), matched, missing, missing_words,
        strengths, weaknesses, recommendation, resume
    )


def score_label(score):
    return "Excellent" if score >= .80 else "Strong" if score >= .65 else \
           "Good" if score >= .45 else "Partial" if score >= .25 else "Low"


def screening_chart(result):
    labels = ["Keywords", "Similarity", "Skills", "Title", "Length"]
    values = [result.keyword_match, result.similarity, result.skill_match,
              result.title_match, result.length_fit]
    fig = go.Figure(go.Bar(
        x=labels, y=[v * 100 for v in values],
        text=[f"{v:.0%}" for v in values], textposition="outside"
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(range=[0, 100], title="Score (%)"),
                      showlegend=False)
    return fig


def skill_chart(result):
    fig = go.Figure(go.Pie(
        labels=["Matched", "Missing"],
        values=[max(len(result.matched_skills), 1), max(len(result.missing_skills), 1)],
        hole=.58
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def make_pdf(result, role):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=14*mm,
                            leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"],
                              fontSize=9, leading=12, textColor=colors.HexColor("#334155")))
    story = [
        Paragraph("AI Resume Screening & Job Matching System", styles["Title"]),
        Paragraph(f"<b>Candidate:</b> {result.name}", styles["Small"]),
        Paragraph(f"<b>Target role:</b> {role}", styles["Small"]),
        Paragraph(f"<b>Generated:</b> {datetime.now():%Y-%m-%d %H:%M}", styles["Small"]),
        Spacer(1, 8),
    ]
    table = Table([
        ["ATS Score", f"{result.score:.0%}", "Recommendation", result.recommendation],
        ["Skill Match", f"{result.skill_match:.0%}", "Keyword Match", f"{result.keyword_match:.0%}"],
        ["Similarity", f"{result.similarity:.0%}", "Title Match", f"{result.title_match:.0%}"],
    ], colWidths=[35*mm, 28*mm, 35*mm, 40*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)

    for heading, items in [
        ("Strengths", result.strengths or ["No major strengths detected."]),
        ("Weaknesses / Gaps", result.weaknesses or ["No major weaknesses detected."]),
        ("Matched Skills", result.matched_skills or ["None"]),
        ("Missing Skills", result.missing_skills or ["None"]),
        ("Missing Keywords", result.missing_keywords or ["None"]),
    ]:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(" • ".join(items), styles["Small"]))

    story.append(Paragraph("Resume Text Snapshot", styles["Heading2"]))
    safe_text = (result.raw_text[:5000] or "No text extracted.")
    safe_text = safe_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(safe_text.replace("\n", "<br/>"), styles["Small"]))
    doc.build(story)
    return buffer.getvalue()


# -------------------------- USER INTERFACE --------------------------

st.markdown("""
<div class="hero">
<h1 style="color:#f8fafc;margin:0">🧠 AI Resume Screening & Job Matching System</h1>
<p class="muted">Screen resumes, compare job skills, rank candidates and download PDF reports.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Recruiter Controls")
    uploads = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    sample = st.toggle("Use sample data", value=True)
    show_rankings = st.toggle("Show ranking table", value=True)
    show_reports = st.toggle("Show PDF downloads", value=True)

left, right = st.columns(2)
with left:
    st.subheader("Job Description")
    job_text = st.text_area("Paste the job description",
                            value=SAMPLE_JOB if sample else "", height=260)
with right:
    st.subheader("Resume Input")
    if uploads:
        selected = st.selectbox("Preview a resume", [f.name for f in uploads])
        selected_file = next(f for f in uploads if f.name == selected)
        preview_text = read_resume(selected_file)
    else:
        preview_text = SAMPLE_RESUME if sample else ""
    resume_text = st.text_area("Resume text", value=preview_text, height=260)

role = job_title(job_text)
if uploads:
    candidates = [(f.name, screen_resume(read_resume(f), job_text)) for f in uploads]
else:
    candidates = [("Sample Candidate", screen_resume(resume_text, job_text))] if resume_text.strip() else []

if not candidates:
    st.info("Upload a resume or turn on sample data to begin screening.")
    st.stop()

ranked = sorted(candidates, key=lambda item: item[1].score, reverse=True)
first = ranked[0][1]

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("ATS Score", f"{first.score:.0%}")
m2.metric("Match Level", score_label(first.score))
m3.metric("Candidates", len(candidates))
m4.metric("Target Role", role)
st.progress(first.score)

dashboard, reports, analysis, preview = st.tabs([
    "Dashboard", "Candidate Reports", "Keyword Analysis", "Resume Preview"
])

with dashboard:
    a, b = st.columns(2)
    a.subheader("Screening Breakdown")
    a.plotly_chart(screening_chart(first), use_container_width=True)
    b.subheader("Skill Coverage")
    b.plotly_chart(skill_chart(first), use_container_width=True)

    st.subheader("Candidate Summary")
    for rank, (name, result) in enumerate(ranked, 1):
        st.markdown(f"""
        <div class="candidate">
            <h4 style="color:#f8fafc;margin:0">{rank}. {name}</h4>
            <p class="muted">ATS Score: <b>{result.score:.0%}</b> |
            {score_label(result.score)} | Recommendation: <b>{result.recommendation}</b></p>
            <p class="muted">Email: {'✓' if result.email else '✗'} |
            Phone: {'✓' if result.phone else '✗'} |
            LinkedIn: {'✓' if result.linkedin else '✗'} |
            GitHub: {'✓' if result.github else '✗'}</p>
        </div>
        """, unsafe_allow_html=True)

with reports:
    if show_rankings and len(ranked) > 1:
        rows = [{
            "Rank": i, "Candidate": name, "ATS Score": round(result.score*100, 1),
            "Recommendation": result.recommendation,
            "Matched Skills": len(result.matched_skills),
            "Missing Skills": len(result.missing_skills),
        } for i, (name, result) in enumerate(ranked, 1)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif len(ranked) == 1:
        st.info("Upload multiple resumes to enable candidate ranking.")

    if show_reports:
        st.subheader("Download PDF Reports")
        for name, result in ranked:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "candidate"
            st.download_button(
                f"Download PDF Report - {name}", make_pdf(result, role),
                file_name=f"{safe_name}_ATS_Report.pdf", mime="application/pdf",
                key=f"pdf_{safe_name}"
            )

with analysis:
    st.subheader("Top Resume Keywords")
    st.dataframe(pd.DataFrame(first.resume_keywords, columns=["Keyword", "Count"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Top Job Keywords")
    st.dataframe(pd.DataFrame(first.job_keywords, columns=["Keyword", "Count"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Missing Keywords")
    st.write(", ".join(first.missing_keywords) or "None")
    st.subheader("Missing Skills")
    st.write(", ".join(first.missing_skills) or "None")

with preview:
    st.subheader("Resume Text Preview")
    st.text_area("Extracted resume content", first.raw_text[:8000], height=300)
    st.subheader("Job Description Preview")
    st.text_area("Job description", job_text[:8000], height=220)

st.caption("This tool supports resume screening. Final hiring decisions must always include human review.")



