import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Sample FAQ data
faqs = {
    "What is your return policy?": "Our return policy lasts 30 days.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "Do you offer international shipping?": "Yes, we ship internationally with additional charges.",
    "How do I reset my password?": "Click on 'Forgot Password' at login to reset your password.",
    "What payment methods do you accept?": "We accept credit cards, debit cards, PayPal, and UPI.",
    "How do I cancel my order?": "You can cancel your order from the 'My Orders' section within 12 hours.",
    "When will my refund be processed?": "Refunds are processed within 5-7 business days after approval.",
    "Do you have a mobile app?": "Yes, our mobile app is available on both iOS and Android platforms.",
    "How can I contact customer support?": "You can contact our support team via the 'Contact Us' page.",
    "Is my personal information secure?": "Yes, we use industry-standard encryption to protect your data.",
    "Do you offer gift cards?": "Yes, gift cards are available in various denominations on our site.",
    "Can I change my shipping address after ordering?": "You can change it within 2 hours of placing the order.",
    "Why was my payment declined?": "Payment may fail due to insufficient balance or incorrect details.",
    "How do I subscribe to your newsletter?": "Scroll to the bottom of the homepage and enter your email.",
    "Are your products eco-friendly?": "We offer a wide range of eco-friendly and sustainable products."
}

# Preprocessing function without NLTK
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Preprocess FAQ questions
faq_questions = list(faqs.keys())
preprocessed_questions = [preprocess(q) for q in faq_questions]

# Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 FAQ Chatbot")
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        border: 2px solid #1f77b4;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("Welcome! Ask me anything about our services, and I’ll help you out.")

user_input = st.text_input("Ask your question here")

if user_input:
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vector, faq_vectors)
    most_similar_idx = similarities.argmax()
    most_similar_score = similarities[0, most_similar_idx]

    if most_similar_score > 0.3:
        response = faqs[faq_questions[most_similar_idx]]
    else:
        response = "Sorry, I couldn't understand your question. Please try again."

    st.markdown(f"**Answer:** {response}")
