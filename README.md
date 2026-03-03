# ResumeIQ — Smart Resume Keyword Analyzer

> **Resume bullet**: Built an NLP-based resume analyzer that computes job-description similarity using TF-IDF and cosine similarity, improving keyword matching automation.

## Features

- 📄 Upload resume as **PDF or plain text**
- 💼 Upload job description as **PDF or plain text**
- 🧠 **TF-IDF + Cosine Similarity** scoring (scikit-learn)
- 🏷️ **Keyword extraction** across 6 tech categories (languages, frameworks, cloud/devops, databases, concepts, tools)
- ❌ **Missing skill suggestions** grouped by category
- 🔍 **Top JD terms not in resume** (TF-IDF based)
- 📊 Three-score dashboard: Overall, Cosine Similarity, Keyword Match %
- ⚡ Fast, clean UI — no page reloads

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | Python, Flask |
| NLP/ML | scikit-learn (TF-IDF, cosine similarity) |
| PDF Parsing | pdfplumber |
| Frontend | Vanilla HTML/CSS/JS |
| Container | Docker, Docker Compose |

## Quick Start

### Option 1: Run directly with Python

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Option 2: Docker (recommended)

```bash
docker-compose up --build
# Open http://localhost:5000
```

## How It Works

1. **Text Extraction** — `pdfplumber` extracts raw text from uploaded PDFs
2. **Keyword Matching** — regex-based scan against a curated bank of 100+ tech keywords across 6 categories
3. **TF-IDF Vectorization** — both documents are vectorized using `TfidfVectorizer` with 1-2 gram range
4. **Cosine Similarity** — measures angular distance between resume and JD vectors (0–100%)
5. **Overall Score** — weighted blend: 60% cosine similarity + 40% keyword match %
6. **Gap Analysis** — highlights keywords present in JD but absent from resume, grouped by category

## API

`POST /analyze`

| Field | Type | Description |
|-------|------|-------------|
| `resume_file` | File | PDF or .txt resume |
| `resume_text` | String | Raw resume text (alternative) |
| `jd_file` | File | PDF or .txt job description |
| `jd_text` | String | Raw JD text (alternative) |

**Response:**
```json
{
  "overall_score": 72.4,
  "similarity_score": 68.1,
  "keyword_match_pct": 80.0,
  "matched_keywords": ["python", "docker", "flask", "..."],
  "missing_keywords": ["kubernetes", "terraform", "..."],
  "missing_by_category": { "cloud_devops": ["kubernetes"], "...": [] },
  "missing_tfidf_terms": ["distributed systems", "ci/cd", "..."]
}
```

## Project Structure

```
resume-analyzer/
├── app.py              # Flask app + NLP logic
├── requirements.txt    # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── templates/
    └── index.html      # Single-file frontend UI
```
