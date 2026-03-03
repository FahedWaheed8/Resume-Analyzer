import os
import re
import json
import pdfplumber
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Technical keyword bank ────────────────────────────────────────────────────
TECH_KEYWORDS = {
    "languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "bash",
        "shell", "html", "css", "sass", "less"
    ],
    "frameworks": [
        "react", "vue", "angular", "django", "flask", "fastapi", "spring", "express",
        "next.js", "nuxt", "svelte", "rails", "laravel", "tensorflow", "pytorch",
        "keras", "sklearn", "scikit-learn", "pandas", "numpy", "scipy", "spark",
        "hadoop", "kafka", "celery", "graphql", "rest", "grpc"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "circleci", "helm", "prometheus", "grafana",
        "nginx", "apache", "linux", "unix", "ci/cd", "devops", "microservices"
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "sqlite", "oracle", "neo4j", "firebase", "snowflake", "bigquery"
    ],
    "concepts": [
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "api", "rest api", "agile", "scrum", "tdd", "oop", "functional programming",
        "distributed systems", "system design", "algorithms", "data structures",
        "security", "encryption", "oauth", "jwt", "websockets", "caching",
        "load balancing", "message queue", "etl", "data pipeline"
    ],
    "tools": [
        "git", "jira", "confluence", "slack", "figma", "tableau", "power bi",
        "airflow", "dbt", "mlflow", "wandb", "hugging face", "openai", "langchain"
    ]
}

ALL_TECH_KEYWORDS = []
for category_keywords in TECH_KEYWORDS.values():
    ALL_TECH_KEYWORDS.extend(category_keywords)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Normalize text for NLP processing."""
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str) -> dict:
    """Extract technical keywords from text, grouped by category."""
    text_lower = text.lower()
    found = {}
    for category, keywords in TECH_KEYWORDS.items():
        matches = []
        for kw in keywords:
            # Use word boundary matching
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(kw)
        if matches:
            found[category] = matches
    return found


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """Compute TF-IDF cosine similarity between resume and job description."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 1)


def get_top_tfidf_terms(text: str, top_n: int = 20) -> list:
    """Extract top TF-IDF terms from a single document."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=1000
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(text)])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except Exception:
        return []


def analyze(resume_text: str, jd_text: str) -> dict:
    """Full analysis pipeline."""
    similarity_score = compute_similarity(resume_text, jd_text)

    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    # Flatten keyword sets
    resume_flat = set()
    for kws in resume_keywords.values():
        resume_flat.update(kws)

    jd_flat = set()
    for kws in jd_keywords.values():
        jd_flat.update(kws)

    matched_keywords = sorted(resume_flat & jd_flat)
    missing_keywords = sorted(jd_flat - resume_flat)

    # Top JD terms not in resume (via TF-IDF)
    jd_top_terms = get_top_tfidf_terms(jd_text, 30)
    resume_top_terms = set(get_top_tfidf_terms(resume_text, 50))
    missing_tfidf_terms = [t for t in jd_top_terms if t not in resume_top_terms][:15]

    # Group missing keywords by category
    missing_by_category = {}
    for category, keywords in TECH_KEYWORDS.items():
        missing = [kw for kw in keywords if kw in missing_keywords]
        if missing:
            missing_by_category[category] = missing

    # Keyword density score
    if jd_flat:
        keyword_match_pct = round(len(matched_keywords) / len(jd_flat) * 100, 1)
    else:
        keyword_match_pct = 0.0

    # Overall score: weighted combo
    overall = round(similarity_score * 0.6 + keyword_match_pct * 0.4, 1)

    return {
        "similarity_score": similarity_score,
        "keyword_match_pct": keyword_match_pct,
        "overall_score": overall,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "missing_by_category": missing_by_category,
        "missing_tfidf_terms": missing_tfidf_terms,
        "resume_keyword_count": len(resume_flat),
        "jd_keyword_count": len(jd_flat),
        "resume_keywords_grouped": resume_keywords,
        "jd_keywords_grouped": jd_keywords,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_route():
    resume_text = ""
    jd_text = ""

    # Handle resume
    if 'resume_file' in request.files and request.files['resume_file'].filename:
        f = request.files['resume_file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(path)
        else:
            resume_text = extract_text_from_txt(path)
        os.remove(path)
    elif request.form.get('resume_text'):
        resume_text = request.form['resume_text']

    if not resume_text.strip():
        return jsonify({"error": "No resume content provided."}), 400

    # Handle job description
    if 'jd_file' in request.files and request.files['jd_file'].filename:
        f = request.files['jd_file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        if filename.lower().endswith('.pdf'):
            jd_text = extract_text_from_pdf(path)
        else:
            jd_text = extract_text_from_txt(path)
        os.remove(path)
    elif request.form.get('jd_text'):
        jd_text = request.form['jd_text']

    if not jd_text.strip():
        return jsonify({"error": "No job description content provided."}), 400

    try:
        result = analyze(resume_text, jd_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
