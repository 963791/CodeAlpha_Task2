import streamlit as st
import nltk
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure nltk data is available
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt")):
    nltk.download("punkt", download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
    nltk.download("stopwords", download_dir=nltk_data_path)

# Sample FAQs
faqs = {
    "What is your return policy?": "You can return any item within 30 days for a full refund.",
    "How do I track my order?": "Use the tracking link sent to your email after the order is shipped.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, and PayPal.",
    "Do you offer customer support?": "Yes, our support team is available 24/7 via chat and email.",
}

# Preprocessing function
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    cleaned = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned)

# Preprocess FAQ questions
faq_questions = list(faqs.keys())
preprocessed_questions = [preprocess(q) for q in faq_questions]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.title("FAQ Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0][best_match_index]
    
    if best_match_score > 0.2:
        st.write("Answer:", faqs[faq_questions[best_match_index]])
    else:
        st.write("Sorry, I couldn't find a matching answer.")
