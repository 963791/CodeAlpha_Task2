import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQs (add your own)
faqs = {
    "What is your return policy?": "Our return policy lasts 30 days.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "Do you offer international shipping?": "Yes, we ship internationally with additional charges.",
    "How do I reset my password?": "Click on 'Forgot Password' at login to reset your password."
}

# Simple preprocessing function without nltk
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Preprocess questions
faq_questions = list(faqs.keys())
preprocessed_questions = [preprocess(q) for q in faq_questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.title("FAQ Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    preprocessed_input = preprocess(user_input)
    input_vector = vectorizer.transform([preprocessed_input])
    similarity_scores = cosine_similarity(input_vector, faq_vectors)
    best_match_idx = similarity_scores.argmax()

    if similarity_scores[0][best_match_idx] > 0.3:
        st.success(f"**Answer:** {faqs[faq_questions[best_match_idx]]}")
    else:
        st.warning("Sorry, I couldn't find a relevant answer.")
