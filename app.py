from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the assessments data
df = pd.read_csv("shl_catalogue.csv")

# Vectorizer
vectorizer = TfidfVectorizer()

@app.route("/")
def index():
    return "SHL Recommendation API is running!"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    tfidf_matrix = vectorizer.fit_transform(df["Description"].tolist() + [query])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    top_indices = cosine_sim.argsort()[-10:][::-1]
    results = []

    for idx in top_indices:
        results.append({
            "Assessment Name": df.iloc[idx]["Assessment Name"],
            "URL": df.iloc[idx]["URL"],
            "Remote Testing Support": df.iloc[idx]["Remote Testing Support"],
            "Adaptive/IRT Support": df.iloc[idx]["Adaptive/IRT Support"],
            "Duration": df.iloc[idx]["Duration"],
            "Test Type": df.iloc[idx]["Test Type"]
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
