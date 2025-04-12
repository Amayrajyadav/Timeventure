import os
import streamlit as st
import re
import base64
import requests
import random
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from gtts import gTTS
from io import BytesIO
from PIL import Image
from deep_translator import GoogleTranslator
from langdetect import detect

# Set page config must be first Streamlit command
st.set_page_config(page_title="Historical Places Explorer", layout="wide", page_icon="🏛️")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key missing.")
    st.stop()
if not HF_API_KEY:
    st.error("⚠️ HuggingFace API key missing.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# Hugging Face client with proper authentication
hf_client = InferenceClient(token=HF_API_KEY)

# Historical places data with proper image URLs
historical_places = {
    "India": {
        "Telangana": {
            "Charminar": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Charminar_Hyderabad_1.jpg/330px-Charminar_Hyderabad_1.jpg",
            "Golconda Fort": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Golconda_Fort_005.jpg/500px-Golconda_Fort_005.jpg",
            "Qutb Shahi Tombs": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Qutb_Shahi_Tomb_5.jpg/500px-Qutb_Shahi_Tomb_5.jpg",
            "Warangal Fort": "https://telanganatourism.gov.in/blog/images/02-08-2019.jpg",
            "Ramoji Film City": "https://hyderabadtourpackage.in/images/places-to-visit/ramoji-film-city-hyderabad-entryfee-timings-tour-package-header.jpg"
        },
        "Maharashtra": {
            "Gateway of India": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Mumbai_03-2016_30_Gateway_of_India.jpg/375px-Mumbai_03-2016_30_Gateway_of_India.jpg",
            "Ajanta Caves": "https://www.pilgrimagetour.in/blog/wp-content/uploads/2024/01/Best-Time-to-Visit-Ajanta-Caves.jpg",
            "Ellora Caves": "https://s7ap1.scene7.com/is/image/incredibleindia/ellora-caves-chhatrapati-sambhaji-nagar-maharashtra-attr-hero-1?qlt=82&ts=1727010540087",
            "Shaniwar Wada": "https://www.savaari.com/blog/wp-content/uploads/2022/11/Shaniwaarwada_Pune_11zon.jpg"
        }
    },
    "USA": {
        "New York": {
            "Statue of Liberty": "https://www.worldatlas.com/r/w1300/upload/f4/d8/7b/shutterstock-1397031029.jpg",
            "Central Park": "https://cdn.prod.website-files.com/5e1f39c11dc59668da99fae2/675c308451319ef384636daa_lowresshutterstock_1414639229-p-2000.jpeg",
            "Empire State Building": "https://www.esbnyc.com/sites/default/files/2025-03/ESB-DarkBlueSky.webp",
            "Brooklyn Bridge": "https://www.nyctourism.com/_next/image/?url=https://images.ctfassets.net/1aemqu6a6t65/68nkexvLlGiTxvxFvzoELk/68ee51265baad76b8d7f5ae8cd99bf2c/brooklyn-bridge-sunset-julienne-schaer.jpg?fm=webp&w=1200&q=75"
        }
    }
}

# Available TTS languages
tts_languages = {
    "Telugu": "te",
    "Hindi": "hi",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt"
}

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stSelectbox, .stTextInput, .stButton>button {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a5a80;
        color: white;
    }
    .place-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        color: #4a6fa5;
        border-bottom: 2px solid #4a6fa5;
        padding-bottom: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .place-image {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def clean_text_for_tts(text):
    """Remove markdown symbols and special characters from text for TTS"""
    # Remove markdown headers
    text = re.sub(r'#+\s*', '', text)
    # Remove markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove other markdown symbols
    text = text.replace('`', '').replace('~', '')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

@st.cache_data(ttl=3600)
def fetch_place_details(place_name, state, country):
    prompt = f"""
    Provide comprehensive details about {place_name} in {state}, {country}:
    - A short historical narrative (150-200 words) in point form
    - Five interesting historical facts
    - Architectural significance (if applicable) in point form
    - Best time to visit 
    - Cultural importance in point form
    - Any special events or festivals associated in point form
    
    Format the response with clear section headings in Markdown.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text if response.text else "No details available."
    except Exception as e:
        return f"⚠️ Error retrieving details: {e}"

@st.cache_data(ttl=3600)
def generate_tts_audio(text, lang='en'):
    try:
        cleaned_text = clean_text_for_tts(text)
        tts = gTTS(text=cleaned_text, lang=lang)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio in {lang}: {e}")
        try:
            if lang != 'en':
                cleaned_text = clean_text_for_tts(text)
                tts = gTTS(text=cleaned_text, lang='en')
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                return audio_bytes
        except:
            pass
        return None

def analyze_sentiment(text):
    try:
        output = hf_client.text_classification(
            text=text, 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        return output[0]['label']
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        return "Neutral"

def translate_text(text, target_lang='hi'):
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        return text
    except Exception as e:
        st.warning(f"Translation error: {e}")
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
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_data = img_byte_arr.getvalue()
            
            try:
                # Try two different approaches for captioning
                try:
                    # Method 1: Direct bytes
                    caption = hf_client.image_to_text(
                        image=img_data,
                        model="nlpconnect/vit-gpt2-image-captioning"
                    )
                    if caption:
                        return caption
                except:
                    # Method 2: Base64 encoded
                    base64_image = base64.b64encode(img_data).decode('utf-8')
                    caption = hf_client.image_to_text(
                        image=base64_image,
                        model="Salesforce/blip-image-captioning-base"
                    )
                    return caption if caption else "No caption could be generated."
                
                return "No caption could be generated for this image."
            except Exception as e:
                st.error(f"Caption generation failed: {str(e)}")
                return "Could not generate caption for this image."
        else:
            return f"Failed to load image. Status code: {response.status_code}"
    except Exception as e:
        return f"Image processing error: {str(e)}"

def analyze_uploaded_image(image_bytes):
    try:
        image = Image.open(image_bytes)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Analyze this historical place image and provide: "
            "1. Likely name and location of the place "
            "2. Brief historical background (100 words) in point form "
            "3. Architectural style and period in point form "
            "4. Cultural significance in point form "
            "5. Best time to visit " 
            "Format your response with clear headings", 
            image
        ])
        return response.text
    except Exception as e:
        return f"⚠️ Error analyzing image: {str(e)}"

def verify_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except:
        return False

# Main app content
st.title("🏛️ Historical Places Explorer")

# Country and state selection
country = st.selectbox("🌍 Select Country", ["Select a country", "India", "USA"])
states_dict = {
    "India": ["Select a state", "Telangana", "Maharashtra"],
    "USA": ["Select a state", "New York"]
}

if country != "Select a country":
    state = st.selectbox("📍 Select State", states_dict[country])

# TTS language selection
default_lang_index = 0
if "selected_tts_language" not in st.session_state:
    st.session_state.selected_tts_language = "Telugu" if country == "India" and state == "Telangana" else "Telugu"

selected_tts_language_name = st.selectbox(
    "🗣️ Select Audio Guide Language", 
    list(tts_languages.keys()),
    index=default_lang_index
)
selected_tts_language_code = tts_languages[selected_tts_language_name]

# Session state to control rendering
if "explore_clicked" not in st.session_state:
    st.session_state.explore_clicked = False
if "selected_places" not in st.session_state:
    st.session_state.selected_places = []

# Buttons to control flow
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗺️ Explore Places") and country != "Select a country" and state != "Select a state":
        st.session_state.explore_clicked = True
        st.session_state.selected_places = random.sample(
            list(historical_places[country][state].keys()), 
            min(3, len(historical_places[country][state])))
with col2:
    if st.button("🔄 Reset"):
        st.session_state.explore_clicked = False
        st.session_state.selected_places = []

# Sidebar for image upload
with st.sidebar:
    st.header("🔍 Image Analysis")
    uploaded_file = st.file_uploader("Upload an image of a historical place", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Analyzing image..."):
            analysis_result = analyze_uploaded_image(uploaded_file)
            st.markdown("### Analysis Results")
            st.markdown(analysis_result)
            
            audio_bytes = generate_tts_audio(analysis_result, selected_tts_language_code)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')

# Main content
if st.session_state.explore_clicked and country != "Select a country" and state != "Select a state":
    st.subheader(f"🔍 Historical Places in {state}, {country}")
    
    for place in st.session_state.selected_places:
        with st.container():
            st.markdown(f"<div class='place-card'>", unsafe_allow_html=True)
            
            st.markdown(f"### 🏰 {place}")
            
            if country in historical_places and state in historical_places[country] and place in historical_places[country][state]:
                image_url = historical_places[country][state][place]
                
                if verify_image_url(image_url):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        try:
                            st.image(image_url, use_container_width=True, caption=place)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                            st.markdown(f"**Image URL:** [Link]({image_url})")
                    with col2:
                        st.markdown("**📸 Caption:**")
                        with st.spinner("Generating caption..."):
                            caption = caption_image(image_url)
                            st.write(caption)
                else:
                    st.warning(f"⚠️ Image for {place} might not be accessible. URL: {image_url}")
            else:
                st.warning("⚠️ No image available for this place.")
            
            with st.expander("📜 View Details"):
                details = fetch_place_details(place, state, country)
                st.markdown(details)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🎧 Audio Guide ({selected_tts_language_name})**")
                    
                    if selected_tts_language_code != "en":
                        translated_text = translate_text(details, selected_tts_language_code)
                        audio_bytes = generate_tts_audio(translated_text, selected_tts_language_code)
                    else:
                        audio_bytes = generate_tts_audio(details, selected_tts_language_code)
                    
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')
                    else:
                        st.warning("⚠️ Audio generation failed")
                
                with col2:
                    if selected_tts_language_code == "te":
                        translation_language = "Hindi"
                        translation_code = "hi"
                    elif selected_tts_language_code == "hi":
                        translation_language = "Telugu"
                        translation_code = "te"
                    else:
                        translation_language = "Telugu" if country == "India" and state == "Telangana" else "Hindi"
                        translation_code = "te" if country == "India" and state == "Telangana" else "hi"
                    
                    st.markdown(f"**🌐 {translation_language} Translation**")
                    st.write(translate_text(details, target_lang=translation_code))
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔍 TL;DR**")
                    st.write(summarize_text(details))
                with col2:
                    sample_review = f"I visited {place} and was amazed by its historical significance!"
                    sentiment = analyze_sentient(sample_review)
                    st.markdown("**🧠 Sample Sentiment Analysis**")
                    st.write(f"`{sentiment}`")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Personalized Recommendations")
    user_feeling = st.text_input("How are you feeling today?", 
                               key="emotion_input",
                               placeholder="Describe your mood...")
    if user_feeling:
        emotion = analyze_sentiment(user_feeling)
        st.write(f"😊 Detected Emotion: `{emotion}`")
        rec = random.choice(st.session_state.selected_places)
        st.success(f"🎯 Based on your mood, we recommend: **{rec}**")
    
    st.markdown("---")
    st.markdown(f"### 🗣️ Local Language Narration ({selected_tts_language_name})")
    if st.session_state.selected_places:
        details = fetch_place_details(st.session_state.selected_places[0], state, country)
        
        if country == "India" and state == "Telangana" and selected_tts_language_code == "en":
            st.info("📢 For Telangana locations, try the Telugu audio option!")
        
        if selected_tts_language_code != "en":
            audio_text = translate_text(details, selected_tts_language_code)
        else:
            audio_text = details
        audio_bytes = generate_tts_audio(audio_text, lang=selected_tts_language_code)
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
        else:
            st.warning(f"⚠️ Could not generate audio in {selected_tts_language_name}")
else:
    st.info("👆 Select a country and state to explore historical places.")
    try:
        st.image("logo.jpg", 
                caption="Explore Historical Places", use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying welcome image: {e}")
        st.markdown("Welcome to the Historical Places Explorer! Select a country and state to begin.")