import os
import streamlit as st
import google.generativeai as genai
import requests
import random
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from gtts import gTTS
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator
import folium
from streamlit_folium import st_folium

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("⚠️ Gemini API key missing.")
if not HF_API_KEY:
    raise ValueError("⚠️ HuggingFace API key missing.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash-latest"

# Hugging Face client
hf_client = InferenceClient(token=HF_API_KEY)

# Coordinates for demo
coordinates = {
    "Golconda Fort": (17.3833, 78.4011),
    "Charminar": (17.3616, 78.4747),
    "Qutb Shahi Tombs": (17.3949, 78.3949),
    "Ramoji Film City": (17.2543, 78.6808),
    "Gateway of India": (18.9218, 72.8347),
    "Ajanta Caves": (20.5522, 75.7033),
    "Ellora Caves": (20.0268, 75.1790),
    "Shaniwar Wada": (18.5196, 73.8553),
    "Statue of Liberty": (40.6892, -74.0445),
    "Central Park": (40.7851, -73.9683),
    "Empire State Building": (40.7484, -73.9857),
    "Brooklyn Bridge": (40.7061, -73.9969)
}

@st.cache_data(ttl=3600)
def fetch_place_details(place_name, state, country):
    prompt = f"""
    Provide details about {place_name} in {state}, {country}:
    - A short historical narrative (100 words).
    - Five interesting historical facts.
    - Traffic info (peak hours, best visit times).
    - Location details.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text if response.text else "No details available."
    except Exception as e:
        return f"⚠️ Error retrieving details: {e}"

@st.cache_data(ttl=3600)
def fetch_wikimedia_image(query):
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "namespace": 0,
        "format": "json"
    }
    try:
        search_resp = requests.get(search_url, params=search_params).json()
        if not search_resp[1]:
            return None
        title = search_resp[1][0]
        image_params = {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "original",
            "titles": title
        }
        image_resp = requests.get(search_url, params=image_params).json()
        pages = image_resp.get("query", {}).get("pages", {})
        for page in pages.values():
            if "original" in page:
                return page["original"]["source"]
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def generate_tts_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def analyze_sentiment(text):
    output = hf_client.text_classification(text, model="cardiffnlp/twitter-roberta-base-sentiment")
    return output[0]['label']

def translate_text(text, target_lang='hi'):
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        return text
    except Exception:
        return text

def summarize_text(text):
    try:
        summary = hf_client.summarization(
            text=text,
            model="facebook/bart-large-cnn"
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"⚠️ Error summarizing text: {e}"

def caption_image(image_url):
    try:
        image = requests.get(image_url).content
        return hf_client.image_to_text(image=image, model="nlpconnect/vit-gpt2-image-captioning")
    except:
        return "No caption available."

def answer_question(context, question):
    try:
        result = hf_client.question_answering(model="deepset/roberta-base-squad2", question=question, context=context)
        return result['answer']
    except:
        return "Sorry, I couldn't find an answer."

# Streamlit app
st.set_page_config(page_title="Historical Places Explorer", layout="wide")
st.title("🏛️ Historical Places Explorer")

country = st.selectbox("🌍 Select Country", [None, "India", "USA"])
states_dict = {
    "India": [None, "Telangana", "Maharashtra"],
    "USA": [None, "New York"]
}
state = st.selectbox("📍 Select State", states_dict[country] if country else [None])

# Session state to control rendering
if "explore_clicked" not in st.session_state:
    st.session_state.explore_clicked = False
if "selected_places" not in st.session_state:
    st.session_state.selected_places = []

# Buttons to control flow
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗺️ Explore Places") and state:
        st.session_state.explore_clicked = True
        st.session_state.selected_places = random.sample(
            {
                "Telangana": ["Golconda Fort", "Charminar", "Qutb Shahi Tombs", "Ramoji Film City"],
                "Maharashtra": ["Gateway of India", "Ajanta Caves", "Ellora Caves", "Shaniwar Wada"],
                "New York": ["Statue of Liberty", "Central Park", "Empire State Building", "Brooklyn Bridge"]
            }.get(state, []), 3
        )
with col2:
    if st.button("🔄 Reset"):
        st.session_state.explore_clicked = False
        st.session_state.selected_places = []

# Main content
if st.session_state.explore_clicked and state:
    st.subheader(f"🔍 Historical Places in {state}, {country}")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)

    for place in st.session_state.selected_places:
        st.markdown(f"### 🏰 {place}")
        image_url = fetch_wikimedia_image(place)

        if image_url:
            st.image(image_url, width=500)
            st.write("📸 Caption:", caption_image(image_url))
        else:
            st.warning("⚠️ No image found.")

        details = fetch_place_details(place, state, country)
        st.write(details)
        st.audio(generate_tts_audio(details), format='audio/mp3')
        st.markdown("**Hindi Translation:**")
        st.write(translate_text(details, target_lang='hi'))
        st.markdown("**🔍 TL;DR:**")
        st.write(summarize_text(details))

        sample_review = f"I visited {place} and loved it! It's breathtaking."
        sentiment = analyze_sentiment(sample_review)
        st.write(f"🧠 Sample Sentiment: {sentiment}")

        st.markdown("**❓ Ask a question about this place:**")
        question = st.text_input(f"e.g., Who built {place}?", key=place)
        if question:
            st.write("💬 Answer:", answer_question(details, question))

        if place in coordinates:
            lat, lon = coordinates[place]
            folium.Marker(location=[lat, lon], popup=place).add_to(m)

    st_folium(m, width=700, height=500)

    st.markdown("---")
    st.markdown("### 💡 Emotion-Based Recommendations")
    user_feeling = st.text_input("Type how you feel (e.g., 'I want peace' or 'I'm nostalgic')", key="emotion_input")
    if user_feeling:
        emotion = analyze_sentiment(user_feeling)
        st.write(f"😊 Detected Emotion: {emotion}")
        rec = random.choice(st.session_state.selected_places)
        st.write(f"🎯 Based on your mood, you might enjoy visiting: {rec}")

    st.markdown("---")
    st.markdown("### 🗣️ Local Language Narration (Hindi)")
    if st.session_state.selected_places:
        hindi_audio = generate_tts_audio(translate_text(details, 'hi'), lang='hi')
        st.audio(hindi_audio, format='audio/mp3')

    st.markdown("---")
    st.markdown("### 🤖 Historical Chatbot")
    user_q = st.text_input("Ask the 16th-century guide something...", key="historical_qa")
    if user_q:
        char_prompt = f"You are a 16th-century historian. Answer like an ancient royal guide. Question: {user_q}"
        chatbot = genai.GenerativeModel(MODEL_NAME)
        char_response = chatbot.generate_content(char_prompt)
        st.write(char_response.text)
