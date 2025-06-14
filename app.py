import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# FAQ data
faqs = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "How do I track my order?": "You can track your order through the 'Track Order' section on our website.",
    "Do you offer international shipping?": "Yes, we ship internationally with applicable shipping fees.",
    "How can I contact customer service?": "You can contact our support team via the 'Contact Us' page or call our toll-free number.",
    "What payment methods are accepted?": "We accept Visa, MasterCard, PayPal, and UPI.",
    "Can I cancel or change my order?": "Orders can be modified or canceled within 2 hours of placement.",
    "Do you offer gift wrapping?": "Yes, gift wrapping is available at checkout for a small additional fee.",
    "Are there any discounts for bulk purchases?": "Yes, we offer discounts on bulk purchases. Contact support for more info.",
    "How do I reset my password?": "Click on 'Forgot Password' at login and follow the instructions.",
    "Is my personal information secure?": "Yes, we use secure encryption to protect your data."
}

# Preprocessing function (without nltk)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[\d]+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

faq_questions = list(faqs.keys())
preprocessed_questions = [preprocess(q) for q in faq_questions]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
    }
    .user-text {
        color: #262730;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .bot-text {
        background-color: #dfe6fd;
        padding: 0.8rem;
        border-radius: 10px;
        color: #1a1a1a;
        font-size: 1.05rem;
    }
    .header {
        color: #4a4a4a;
        text-align: center;
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>💬 Smart FAQ Chatbot</div>", unsafe_allow_html=True)

# User input
user_question = st.text_input("Ask me a question:", placeholder="e.g., How do I reset my password?", key="input")

if user_question:
    user_preprocessed = preprocess(user_question)
    user_vector = vectorizer.transform([user_preprocessed])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0][best_match_index]

    if best_match_score > 0.2:
        response = faqs[faq_questions[best_match_index]]
    else:
        response = "I'm sorry, I couldn't find an answer to that question. Please contact support."

    st.markdown(f"<div class='user-text'>You: {user_question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-text'>🤖 {response}</div>", unsafe_allow_html=True)

