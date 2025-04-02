from flask import Flask, request, jsonify
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import re
from collections import Counter

app = Flask(__name__)

# 📄 Extract text from PDF file
def extract_text_from_pdf(file_stream):
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text.strip()

# 🧠 Extract basic keywords from text (very simplified)
def extract_keywords(text, top_n=10):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    words = text.lower().split()
    common_words = set([
        'the', 'and', 'for', 'with', 'you', 'your', 'have', 'has', 'are',
        'this', 'that', 'from', 'they', 'will', 'can', 'not', 'use', 'using',
        'all', 'any', 'their', 'which', 'each', 'when', 'how', 'what', 'where'
    ])
    filtered_words = [w for w in words if w not in common_words and len(w) > 2]
    word_freq = Counter(filtered_words)
    return [word for word, _ in word_freq.most_common(top_n)]

# 📍 Endpoint 1: Analyze Resume
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'Resume file is required'}), 400

    resume_file = request.files['resume']
    if resume_file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(BytesIO(resume_file.read()))
    else:
        text = resume_file.read().decode('utf-8')

    keywords = extract_keywords(text)

    return jsonify({
        'text_summary': text[:500] + '...' if len(text) > 500 else text,
        'keywords': keywords
    })

# 📍 Endpoint 2: Match Resume with Job Description
@app.route('/match_job', methods=['POST'])
def match_job():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Resume and job description are required'}), 400

    resume_file = request.files['resume']
    job_desc = request.form['job_description']

    if resume_file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(BytesIO(resume_file.read()))
    else:
        resume_text = resume_file.read().decode('utf-8')

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return jsonify({
        'match_score': round(score * 100, 2),
        'message': 'Higher score means better resume-job fit'
    })

# 🔍 Health check route
@app.route('/')
def home():
    return jsonify({"message": "AI Resume Analyzer API is running (no model download)."})

# 🚀 Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
