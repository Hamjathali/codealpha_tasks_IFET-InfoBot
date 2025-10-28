import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# Safely download NLTK resources on Streamlit Cloud
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Required NLTK packages
required_packages = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"), 
    ("corpora/stopwords", "stopwords")
]

# Download if missing
for path, pkg in required_packages:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)


# DATA LOADING
@st.cache_data
def load_faq(csv_path):
    df = pd.read_csv(csv_path)
    return df['question'].tolist(), df['answer'].tolist()



# TEXT PREPROCESSING
def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)



# BUILD VECTOR MODEL
@st.cache_data
def build_vectorizer(cleaned_questions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_questions)
    return vectorizer, tfidf_matrix



# GET BOT RESPONSE
def get_response(user_input, vectorizer, tfidf_matrix, questions, answers, threshold=0.2):
    user_input_cleaned = preprocess_text(user_input)

    
    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
    if any(greet in user_input_cleaned for greet in greetings):
        return "Hello! 👋 I'm the IFET College InfoBot. How can I help you today?", 1.0

    user_vec = vectorizer.transform([user_input_cleaned])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    best_index = similarity_scores.argsort()[::-1][0]
    best_score = similarity_scores[best_index]

    if best_score < threshold:
        return "I'm not sure about that. Please visit the IFET website or contact the office.", best_score
    else:
        return answers[best_index], best_score



# MAIN STREAMLIT APP
def main():
    st.set_page_config(page_title="IFET College InfoBot", page_icon="🎓", layout="centered")
    st.title("🎓 IFET College InfoBot")
    st.markdown("Ask anything about **IFET College** — courses, admissions, hostel, placements, or faculty!")

    csv_path = "faq_dataset.csv"
    questions, answers = load_faq(csv_path)
    cleaned_questions = [preprocess_text(q) for q in questions]
    vectorizer, tfidf_matrix = build_vectorizer(cleaned_questions)


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None


    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("👤 You:", placeholder="Type your question here...")
        submit = st.form_submit_button("Ask")


    if submit and user_input:
        response, score = get_response(user_input, vectorizer, tfidf_matrix, questions, answers)

        if st.session_state.current_chat:
            st.session_state.chat_history.insert(0, st.session_state.current_chat)
            st.session_state.chat_history = st.session_state.chat_history[:3]

        st.session_state.current_chat = (user_input, response)

    # DISPLAY (STYLED SECTIONS)
    st.markdown(
        """
        <style>
            .chat-user {
                font-size: 20px;
                color: #00BFFF;
                font-weight: bold;
            }
            .chat-bot {
                font-size: 19px;
                color: #FFFFFF;
                background-color: #1E1E1E;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .chat-divider {
                border-top: 1px solid #444;
                margin: 10px 0;
            }
            h3 {
                font-size: 24px !important;
                color: #FFD700;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ======= CURRENT CONVERSATION =======
    if st.session_state.current_chat:
        user_msg, bot_msg = st.session_state.current_chat
        st.markdown("### 💡 Current Conversation")
        st.markdown(f"<p class='chat-user'>👤 You: {user_msg}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bot'>🤖 InfoBot: {bot_msg}</div>", unsafe_allow_html=True)
        st.markdown("<div class='chat-divider'></div>", unsafe_allow_html=True)

    # ======= RECENT CONVERSATIONS =======
    if st.session_state.chat_history:
        st.markdown("### 💬 Recent Conversations")
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"<p class='chat-user'>👤 You: {user_msg}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bot'>🤖 InfoBot: {bot_msg}</div>", unsafe_allow_html=True)
            st.markdown("<div class='chat-divider'></div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <p style="font-size:20px; margin-top:0;">
            👨‍💻 Developed by: <b>Hamjathali I</b>
        </p>
        <p style="font-size:20px;">💡 Idea: <i>🎓 IFET College InfoBot</i></p>
        <p style="font-size:20px;">🛠️ Tech Stack: Python, Streamlit, NLTK, Scikit-learn, Pandas</p>
        """,
        unsafe_allow_html=True
    )



# RUN APP
if __name__ == "__main__":
    main()
