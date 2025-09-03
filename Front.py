import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from backend import analyze_all_clips, chunk_video, create_final_teaser, download_video, select_clips,extract_audio
import json
# Simple replacements
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi", "mkv"]
MAX_FILE_SIZE_MB = 500

def validate_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def generate_caption(tone: str) -> str:
    # Placeholder – can use GroqLLM if you want AI captions
    return f"Here’s a {tone.lower()} teaser for our latest video!"

# Page configuration
st.set_page_config(
    page_title="AI Video Teaser Generator",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load external CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to internal CSS if external file is not found
        st.markdown("""
        <style>
            /* Basic fallback styles */
            .stApp {
                font-family: 'Inter', sans-serif;
                background: #ffffff;
                color: #1a1a1a;
            }
            .app-title {
                font-size: 2.8rem;
                font-weight: 700;
                margin-bottom: 1rem;
                color: #1a1a1a;
            }
            .app-subtitle {
                font-size: 1.2rem;
                color: #666;
                margin-bottom: 2rem;
            }
            .stButton > button {
                background: #4CAF50;
                color: white;
                padding: 14px 28px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            .feature-grid {
                display: flex;
                justify-content: space-between;
                margin: 3rem 0;
            }
            .feature-card {
                text-align: center;
                padding: 1.5rem;
                flex: 1;
                margin: 0 1rem;
            }
            .divider {
                height: 1px;
                background: linear-gradient(to right, transparent, #e0e0e0, transparent);
                margin: 2rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "current_step" not in st.session_state:
        st.session_state.current_step = "welcome"
    
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
        
    if "duration" not in st.session_state:
        st.session_state.duration = 30
        
    if "tone" not in st.session_state:
        st.session_state.tone = "Professional"
    
    if "teaser_path" not in st.session_state:
        st.session_state.teaser_path = None
        
    if "caption" not in st.session_state:
        st.session_state.caption = None
        
    if "add_subtitles" not in st.session_state:
        st.session_state.add_subtitles = True
        
    if "add_music" not in st.session_state:
        st.session_state.add_music = True

    if "analysis" not in st.session_state:
        st.session_state.analysis = None

# Welcome section
def show_welcome():
    # Header
    st.markdown("<h1 class='app-title'>AI Video Teaser Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Transform your videos into compelling teasers with AI. Upload your content, customize preferences, and create engaging previews in minutes.</p>", unsafe_allow_html=True)
    
    # Divider
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Features section using Streamlit columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #4CAF50;'>🤖</div>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; color: #1a1a1a;'>AI-Powered</h3>
            <p style='color: #666; font-size: 0.95rem;'>Smart content analysis and editing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #4CAF50;'>⚡</div>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; color: #1a1a1a;'>Fast Processing</h3>
            <p style='color: #666; font-size: 0.95rem;'>Generate teasers in 3 minutes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #4CAF50;'>🎬</div>
            <h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; color: #1a1a1a;'>Multiple Formats</h3>
            <p style='color: #666; font-size: 0.95rem;'>Custom teaser options</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Divider
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("<h2 style='text-align: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; color: #1a1a1a;'>Get Started</h2>", unsafe_allow_html=True)
    
    # CTA Bullets
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("<p style='text-align: center; color: #4CAF50; font-weight: 500;'>✓ No signup required</p>", unsafe_allow_html=True)
    with col5:
        st.markdown("<p style='text-align: center; color: #4CAF50; font-weight: 500;'>✓ Free to try</p>", unsafe_allow_html=True)
    with col6:
        st.markdown("<p style='text-align: center; color: #4CAF50; font-weight: 500;'>✓ Professional results</p>", unsafe_allow_html=True)
    
    # CTA Button
    if st.button("Create Your Teaser Now", key="start_creation", use_container_width=True):
        st.session_state.current_step = "video_input"
        st.rerun()

# Video input section
def handle_video_input():
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; color: #1a1a1a;'>Upload Your Video</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Provide your video content to get started</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 1px solid rgba(0, 0, 0, 0.05);'>
        """, unsafe_allow_html=True)
        
        st.header("Step 1: Provide Your Video")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload a video file", "Paste YouTube URL"],
            horizontal=True,
            key="input_method"
        )
        
        video_source = None
        
        if input_method == "Upload a video file":
            uploaded_file = st.file_uploader(
                f"Upload your video ({', '.join(SUPPORTED_VIDEO_FORMATS)})", 
                type=SUPPORTED_VIDEO_FORMATS,
                help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
            )
            if uploaded_file is not None:
                # Check file size
                if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.error(f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB")
                else:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        video_source = tmp_file.name
                        st.session_state.video_path = video_source
                        st.success("Video uploaded successfully!")
                    
        else:
            youtube_url = st.text_input(
                "Paste YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_url"
            )
            if youtube_url:
                if validate_youtube_url(youtube_url):
                    video_source = youtube_url
                    st.session_state.video_path = youtube_url
                    st.success("YouTube URL accepted!")
                else:
                    st.error("Please enter a valid YouTube URL")
        
        if video_source:
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Continue to Preferences →", key="to_prefs", use_container_width=True):
                st.session_state.current_step = "preferences"
                st.rerun()
        else:
            st.markdown("</div>", unsafe_allow_html=True)

# Preferences section
def get_user_preferences():
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; color: #1a1a1a;'>Customize Your Teaser</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Adjust settings to match your preferences</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 1px solid rgba(0, 0, 0, 0.05);'>
        """, unsafe_allow_html=True)
        
        st.header("Step 2: Teaser Preferences")
        
        col1 = st.columns(1)[0]
        
        with col1:
            st.subheader("Teaser Specifications")
            duration = st.selectbox(
                "Teaser duration:",
                ["30 seconds", "60 seconds", "Custom"],
                key="duration_select"
            )
            
            if duration == "Custom":
                custom_duration = st.slider("Custom duration (seconds):", 10, 120, 30, key="custom_dur")
                st.session_state.duration = custom_duration
            else:
                st.session_state.duration = int(duration.split()[0])
                
            tone = st.selectbox(
                "Tone:",
                ["Professional", "Exciting", "Educational", "Inspirational", "Compelling"],
                key="tone_select"
            )
            st.session_state.tone = tone
            
       # with col2:
        #    st.subheader("Branding Options")
            
         #   use_branding = st.checkbox("Add branding elements", key="use_branding")
            
          #  if use_branding:
           #     logo = st.file_uploader("Upload logo (optional):", type=["png", "jpg", "jpeg"], key="logo_upload")
            #    if logo:
             #       st.session_state.logo = logo
                    
              #  tagline = st.text_input("Tagline (optional):", key="tagline_input")
               # if tagline:
                #    st.session_state.tagline = tagline
        
        
        # Update session state only when the button is clicked
        if st.button("Generate Teaser →", key="generate_btn", use_container_width=True):
            st.session_state.current_step = "processing"
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

########### Groq LLM wrapper (optional) ###########
class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 800, api_key: str = None):
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            raise RuntimeError("langchain_groq not installed or import failed.") from e

        groq_api_key = "YOUR_GROQ_KEY" or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing. Set it in env or pass as argument.")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(self, system: str, user: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=user))
        resp = self.llm.invoke(msgs)
        return str(resp.content).strip().replace("\n", " ")

# Processing section
def process_video():
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <div style='border: 4px solid #f3f3f3; border-top: 4px solid #4CAF50; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 1.5rem;'></div>
        <h2>Generating Your Teaser</h2>
        <p>Our AI is analyzing your video and creating a compelling teaser...</p>
    </div>
    
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Download YouTube video if needed
    if st.session_state.video_path and ("youtube.com" in st.session_state.video_path or "youtu.be" in st.session_state.video_path):
        status_text.text("Downloading YouTube video...")
        # try:
        #     video_path = download_youtube_video(st.session_state.video_path)
        #     st.session_state.video_path = video_path
        #     progress_bar.progress(20)
        # except Exception as e:
        #     print(f"Error downloading YouTube video: {str(e)}")
        #     st.error(f"Error downloading YouTube video: {str(e)}")
        #     # st.session_state.current_step = "video_input"
        #     st.rerun()
        #     return
    
    # Analyze video content
    status_text.text("Analyzing your video to find key highlights...")
    # try:
    #     highlights = analyze_video_content(
    #         st.session_state.video_path, 
    #         st.session_state.duration, 
    #         st.session_state.tone
    #     )
    #     progress_bar.progress(40)
    # except Exception as e:
    #     st.error(f"Error analyzing video: {str(e)}")
    #     st.session_state.current_step = "video_input"
    #     st.rerun()
    #     return
    
    # Generate teaser
    status_text.text("Creating your teaser...")
    try:
        video_path = download_video(st.session_state.video_path)
        chunks = chunk_video(video_path)

        metadata = analyze_all_clips(chunks, video_path)
        with open("video_analysis.json", "w") as f:
            json.dump(metadata, f, indent=2)
        st.session_state.analysis = metadata
        selected = select_clips(metadata)
        if not selected:
            print("⚠️ No clips selected. Falling back.")
            selected = metadata[:2]
        
        teaser = create_final_teaser(video_path, selected,add_subtitles=st.session_state.add_subtitles, target_length=st.session_state.duration)
        print("✅ Teaser ready:", teaser)
        st.session_state.teaser_path = teaser
        progress_bar.progress(80)
    except Exception as e:
        st.error(f"Error generating teaser: {str(e)}")
        st.session_state.current_step = "video_input"
        
        st.rerun()
        return
    
    # Generate caption
    status_text.text("Generating subtitles")
    try:
        caption = generate_caption(st.session_state.tone)
        st.session_state.caption = caption
        progress_bar.progress(100)
    except Exception as e:
        st.warning(f"Couldn't generate caption: {str(e)}")
        st.session_state.caption = "Check out this teaser for our latest content!"
    
    time.sleep(1)
    
    st.session_state.current_step = "output"
    st.rerun()

# Output section
def show_output_options():
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; color: #1a1a1a;'>Your Teaser is Ready!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Download and share your professionally crafted video teaser</p>", unsafe_allow_html=True)
    
    if st.session_state.teaser_path and os.path.exists(st.session_state.teaser_path):
        with st.container():
            st.markdown("""
            <div style='background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); border: 1px solid rgba(0, 0, 0, 0.05);'>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #4CAF50; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; display: inline-block; margin-bottom: 1.5rem;'>Done! How does this look?</div>
            """, unsafe_allow_html=True)
            
            # Display the generated teaser
            st.video(st.session_state.teaser_path)
            
            col1 = st.columns(1)[0]
            
            with col1:
                st.subheader("Download Options")
                with open(st.session_state.teaser_path, "rb") as file:
                    st.download_button(
                        label="Download Teaser",
                        data=file,
                        file_name="ai_teaser.mp4",
                        mime="video/mp4",
                        key="download_teaser",
                        use_container_width=True
                    )
                if "teaser_path" in st.session_state:
                    video_path = st.session_state.teaser_path

                    # Extract audio
                    audio_path = extract_audio(video_path)

                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, "rb") as f:
                            st.download_button(
                                label="Download Teaser Audio 🎵",
                                data=f,
                                file_name=os.path.basename(audio_path),
                                mime="audio/mpeg",
                                use_container_width=True
                            )
                # Main/original video audio
                if "video_path" in st.session_state:
                    video_path = st.session_state.video_path
                    main_audio = extract_audio(video_path)
                    if main_audio and os.path.exists(main_audio):
                        with open(main_audio, "rb") as f:
                            st.download_button(
                                label="Download Full Video Audio 🎵",
                                data=f,
                                file_name=os.path.basename(main_audio),
                                mime="audio/mpeg",
                                use_container_width=True
                            )    
                

                    
            #with col2:
             #   st.subheader("Additional Options")
                
                # Use unique keys for these widgets
              #  music_option = st.checkbox("Add background music", value=st.session_state.add_music, key="music_option_checkbox")
               # if music_option:
                #    music_style = st.selectbox("Music style:", ["Upbeat", "Corporate", "Inspiring", "Neutral"], key="music_style_select")
                    
                #subs_option = st.checkbox("Customize subtitles", value=st.session_state.add_subtitles, key="subs_option_checkbox")
                #if subs_option:
                 #   font_size = st.slider("Subtitle font size:", 10, 30, 20, key="font_size_slider")
                  #  color = st.color_picker("Subtitle color:", "#FFFFFF", key="sub_color_picker")
            
            #st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Teaser file not found. Please try again.")
        # --- Chatbot Section ---
    st.markdown("### 💬 LLM Video Assistant")
    st.write("Chat with the assistant about your generated video (summaries, hooks, insights)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render conversation as chat bubbles
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div style='text-align: right; margin: 0.5rem 0;'>
                <div style='display: inline-block; background: #DCF8C6; color: #000; 
                            padding: 0.6rem 1rem; border-radius: 12px; max-width: 70%;'>
                    {text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:  # assistant
            st.markdown(f"""
            <div style='text-align: left; margin: 0.5rem 0;'>
                <div style='display: inline-block; background: #E8E8E8; color: #000; 
                            padding: 0.6rem 1rem; border-radius: 12px; max-width: 70%;'>
                    {text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Input at bottom
    user_query = st.text_input("Type your question:", key="chat_input")
    if st.button("Send", key="chat_send"):
        if user_query.strip():
            # Add user message
            st.session_state.chat_history.append(("user", user_query))

            if not st.session_state.analysis:
                st.session_state.chat_history.append(("assistant", "⚠️ No analysis present."))
            else:
                try:
                    llm = GroqLLM()
                    with open("video_analysis.json") as f:
                        analysis_text = f.read()
                    if len(analysis_text.split()) > 3000:
                        analysis_text = " ".join(analysis_text.split()[:3000])

                    system_prompt = "You are an assistant that answers questions about the video metadata and transcript."
                    res = llm.chat(system_prompt, f"Video analysis JSON:\n{analysis_text}\n\nQuestion: {user_query}")

                    # Add assistant response
                    st.session_state.chat_history.append(("assistant", res))
                except Exception as e:
                    st.session_state.chat_history.append(("assistant", f"⚠️ Error: {e}"))

            st.rerun()


    st.markdown("---")
    st.write("Want to create another teaser?")
    if st.button("Start Over", key="restart_btn", use_container_width=True):
        # Clean up temporary files
        from src.utils import cleanup_temp_files
        cleanup_temp_files()
        # Reset session state
        for key in list(st.session_state.keys()):
            if key != "current_step":
                del st.session_state[key]
        st.session_state.current_step = "welcome"
        st.rerun()

# Main application logic
def main():
    # Load custom CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Display the appropriate section based on current step
    if st.session_state.current_step == "welcome":
        show_welcome()
        
    elif st.session_state.current_step == "video_input":
        handle_video_input()
        
    elif st.session_state.current_step == "preferences":
        get_user_preferences()
        
    elif st.session_state.current_step == "processing":
        process_video()
        
    elif st.session_state.current_step == "output":
        show_output_options()

if __name__ == "__main__":

    main()
