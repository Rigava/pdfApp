import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel 
import os 
from groq_translation import groq_translate
from gtts import gTTS 
# Set page config
st.set_page_config(page_title='Groq Translator', page_icon='🎤')

# Set page title
st.title('Groq Translator')

languages = {
   "Portuguese": "pt",
   "Spanish": "es",
   "German": "de",
   "French": "fr",
   "Italian": "it",
   "Dutch": "nl",
   "Russian": "ru",
   "Japanese": "ja",
   "Chinese": "zh",
   "Korean": "ko"
}

# Language selection
option = st.selectbox(
   "Language to translate to:",
   languages,
   index=None,
   placeholder="Select language...",
)
# Load whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2)) # [!code ++]

# Speech to text
def speech_to_text(audio_chunk): # [!code ++]
    segments, info = model.transcribe(audio_chunk, beam_size=5) # [!code ++]
    speech_text = " ".join([segment.text for segment in segments]) # [!code ++]
    return speech_text # [!code ++]
# Text to speech
def text_to_speech(translated_text, language): # [!code ++]
    file_name = "speech.mp3" # [!code ++]
    my_obj = gTTS(text=translated_text, lang=language) # [!code ++]
    my_obj.save(file_name) # [!code ++]
    return file_name # [!code ++]
# Record audio
audio_bytes = audio_recorder()
if audio_bytes and option:
    # Display audio player
    st.audio(audio_bytes, format="audio/wav")

    # Save audio to file
    with open('audio.wav', mode='wb') as f:
        f.write(audio_bytes)


    # Speech to text
    st.divider() # [!code ++]
    with st.spinner('Transcribing...'): # [!code ++]
        text = speech_to_text('audio.wav') # [!code ++]
    st.subheader('Transcribed Text') # [!code ++]
    st.write(text) # [!code ++]
    # Groq translation
    st.divider() # [!code ++]
    with st.spinner('Translating...'): # [!code ++]
        translation = groq_translate(text, 'en', option) # [!code ++]
    st.subheader('Translated Text to ' + option) # [!code ++]
    st.write(translation.text) # [!code ++]
    # Text to speech
    audio_file = text_to_speech(translation.text, languages[option]) # [!code ++]
    st.audio(audio_file, format="audio/mp3")  # [!code ++]
