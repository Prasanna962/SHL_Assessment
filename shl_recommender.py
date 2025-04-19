import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Sample SHL catalog data
data = {
    'Assessment': ['Cognitive Ability Test', 'Personality Test', 'Data Science Test'],
    'Description': ['Measures problem-solving and logical reasoning',
                    'Assesses personality traits and behavior patterns',
                    'Tests knowledge in data science, including Python, statistics'],
    'URL': ['https://www.shl.com/assessments/cognitive-ability',
            'https://www.shl.com/assessments/personality',
            'https://www.shl.com/assessments/data-science'],
    'Remote Testing Support': ['Yes', 'Yes', 'No'],
    'Adaptive/IRT Support': ['Yes', 'No', 'Yes'],
    'Duration': ['40 minutes', '30 minutes', '60 minutes'],
    'Test Type': ['Aptitude', 'Personality', 'Technical']
}

df = pd.DataFrame(data)


# Function to recommend assessments based on user query
def recommend_assessments(query):
    vectorizer = TfidfVectorizer(stop_words='english')
    # Combine the query with the description column of the dataset
    descriptions = df['Description']
    combined = pd.concat([descriptions, pd.Series([query])], ignore_index=True)  # Correcting append using concat

    tfidf_matrix = vectorizer.fit_transform(combined)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare the last entry (query) to the rest

    # Get top 3 recommendations
    recommended_idx = cosine_sim.argsort()[0][-3:][::-1]
    recommendations = df.iloc[recommended_idx]

    return recommendations


# Streamlit UI setup
st.title('SHL Assessment Recommendation System')

# Input from the user
user_input = st.text_area('Enter Job Description or Query', '')

# Button to trigger recommendation
if st.button('Get Recommendations'):
    if user_input:
        recommendations = recommend_assessments(user_input)

        st.write("Recommended Assessments:")
        st.write(recommendations[
                     ['Assessment', 'URL', 'Remote Testing Support', 'Adaptive/IRT Support', 'Duration', 'Test Type']])
    else:
        st.write("Please enter a job description or query to get recommendations.")
