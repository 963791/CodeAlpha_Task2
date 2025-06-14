import json
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load FAQs from JSON
with open("faqs.json", "r") as f:
    faqs = json.load(f)

# Preprocess function
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word not in stop_words and word not in string.punctuation])

# Preprocess FAQ questions
faq_questions = list(faqs.keys())
preprocessed_questions = [preprocess(q) for q in faq_questions]

# Vectorize
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.title("🤖 FAQ Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(input_vector, faq_vectors)
    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_idx]

    if best_score > 0.3:
        st.success(f"Answer: {faqs[faq_questions[best_match_idx]]}")
    else:
        st.warning("Sorry, I don't understand your question.")
