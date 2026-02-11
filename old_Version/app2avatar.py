import streamlit as st
import os
from openai import OpenAI
import tiktoken
import copy 
import streamlit.components.v1 as components 

# --- 1. Configuration and Initialization ---
# Ensure your API Key is set correctly
# OPENAI_API_KEY_2 = "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"
api_key_chatbot = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_2)
try:
    client = OpenAI(api_key=api_key_chatbot)
except Exception as e:
    st.error(f"Failed to initialize OpenAI Client: {e}")
    st.stop()

MODEL = "gpt-4o-mini"
TOKEN_BUDGET = 500
TEMPERATURE = 0.7
MAX_TOKENS = 500
SYSTEM_PROMPT_EMPATHY = (
    "You are Sophia. Act as a supportive psychology teacher. "
    "Learning topics include: Classical Conditioning (Pavlov), Memory Types (Short-term vs Long-term), Cognitive Biases (Confirmation Bias, Availability Heuristic), Motivation Theory (Maslow). "
    "Observe the student's emotions before teaching. Use a kind tone. "
    "Praise effort. Tailor explanations with vivid examples."
    "If the users not understand, you can give them some examples, for example for Classical Conditioning: A dog hears a bell (neutral stimulus) right before receiving food (unconditioned stimulus). Over time, the bell alone makes the dog salivate."
    "You need to ask them some questions actively, such as:"
    "In Pavlov’s experiment, what was the conditioned stimulus?"
    "A. Food, B. Bell, C. Salivation,Correct answer: B. Bell Afterwards giving encouraging, friendly Feedback: "
    "If correct: Correct. The bell became the conditioned stimulus after being associated with food."
    "If incorrect: Not quite. The conditioned stimulus was the bell."
)
SYSTEM_PROMPT_NEUTRAL = (
    "Act as a logical psychology teacher. If the student asks non-psychology topics, respond with "
    "'Don't ask unrelated questions.' Deliver factual definitions only. Ignore emotions."
    "Learning topics include: Classical Conditioning (Pavlov), Memory Types (Short-term vs Long-term), Cognitive Biases (Confirmation Bias, Availability Heuristic), Motivation Theory (Maslow). "
    "You need to ask them some questions actively, such as:"
    "In Pavlov’s experiment, what was the conditioned stimulus?"
    "A. Food, B. Bell, C. Salivation, Correct answer: B. Bell Afterwards giving Feedback: If correct → “Correct.” If incorrect → “Incorrect. The correct answer is: bell."
)


# --- 2. Sentiment Control and Token Functions (Unchanged) ---
POSITIVE_WORDS = ["good", "great", "excellent", "fantastic", "amazing", "wonderful", "positive", "it's possible", "i believe in myself"]
NEGATIVE_WORDS = ["bad", "terrible", "awful", "horrible", "negative", "discouraging", "unsupportive", "unhelpful", "i can't", "i don't", "impossible", "give up"]

class SafeCounter:
    def __init__(self, min_val=-10, max_val=10):
        self.value = 0
        self.min_val = min_val
        self.max_val = max_val

    def increment(self, amount=1):
        self.value = min(self.max_val, self.value + amount)

    def decrement(self, amount=1):
        self.value = max(self.min_val, self.value - amount)

    def reset(self):
        self.value = 0

try:
    ENCODING = tiktoken.encoding_for_model(MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try:
        if not messages or len(messages) <= 1:
            return 0
        return sum(count_tokens(m["content"]) for m in messages[1:]) 
    except:
        return 0

def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    while total_tokens_used(messages) > budget:
        if len(messages) <= 3:
            break
        messages.pop(1)
        messages.pop(1)
    return messages

def detect_sentiment(user_message):
    msg = user_message.lower()
    counter = st.session_state.sentiment_counter
    for w in POSITIVE_WORDS:
        if w in msg:
            counter.increment()
    for w in NEGATIVE_WORDS:
        if w in msg:
            counter.decrement()

def get_current_messages():
    return st.session_state.messages_empathy if st.session_state.empathy_mode else st.session_state.messages_neutral

# --- 3. Chat Logic (Unchanged) ---
def chat_with_chatbot(user_message: str):
    is_empathy = st.session_state.empathy_mode
    current_messages = get_current_messages()

    st.session_state.display_history.append({"role": "user", "content": user_message})
    
    if is_empathy:
        detect_sentiment(user_message)
        
        injected_message = user_message
        counter = st.session_state.sentiment_counter
        
        if counter.value <= -2:
            injected_message = "The student is discouraged. Provide the encouragement, and switch to a simpler topic."
            counter.reset()
        elif counter.value >= 2:
            injected_message = "The student is positive. Acknowledge the enthusiasm, and switch to more advanced concepts."
            counter.reset()
            
        current_messages.append({"role": "user", "content": injected_message})
    else:
        current_messages.append({"role": "user", "content": user_message})

    current_messages = enforce_token_budget(current_messages)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"API Error: {e}"

    current_messages.append({"role": "assistant", "content": reply})
    st.session_state.display_history.append({"role": "assistant", "content": reply})


# --- 4. Session State & UI Handlers ---

def initialize_session_state():
    if "sentiment_counter" not in st.session_state:
        st.session_state.sentiment_counter = SafeCounter()
    if "empathy_mode" not in st.session_state:
        st.session_state.empathy_mode = True 
    if "messages_empathy" not in st.session_state:
        st.session_state.messages_empathy = [{"role": "system", "content": SYSTEM_PROMPT_EMPATHY}]
    if "messages_neutral" not in st.session_state:
        st.session_state.messages_neutral = [{"role": "system", "content": SYSTEM_PROMPT_NEUTRAL}]
    if "display_history" not in st.session_state:
        st.session_state.display_history = []

def on_mode_change():
    st.session_state.sentiment_counter.reset()
    st.session_state.display_history = []


# --- 5. Streamlit UI Layout (Fully English) ---

# --- 3D GLB Renderer Setup ---https://modelviewer.dev/shared-assets/models/TyrannosaurusRex.glb
# 🚨 IMPORTANT: Use a known working GLB URL for testing. 
# REPLACE THIS WITH YOUR OWN PUBLIC READY PLAYER ME MODEL URL.
# = "https://raw.githubusercontent.com/yiyangDelix/readyPlayerMe/main/6903635663bf032571ed7873.glb" 
YOUR_GLB_URL = "https://raw.githubusercontent.com/yiyangDelix/readyPlayerMe/main/6903635663bf032571ed7873.glb" 
# YOUR_GLB_URL = "https://raw.githubusercontent.com/yiyangDelix/readyPlayerMe/main/Talking.glb" 
GLB_VIEWER_HTML = f"""
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>

<model-viewer 
    src="{YOUR_GLB_URL}" 
    alt="Sophia 3D Avatar"
    shadow-intensity="1" 
    camera-controls 
    auto-rotate 
    ar 
    style="width: 100%; height: 500px;"
    exposure="1.0"
    interaction-prompt="none"
>
</model-viewer>
"""
AVATAR_USER = "user" 
AVATAR_SOPHIA = "👩‍🏫" 

initialize_session_state()

st.set_page_config(page_title="Sophia 3D Virtual Psychology Teacher", layout="wide")
st.title("🧠 Sophia 3D Virtual Psychology Teacher Chatbot")

col_avatar, col_chat = st.columns([1, 2]) 

# --- Left Column: 3D Avatar Image ---
with col_avatar:
    st.subheader("Sophia (3D Virtual Avatar)")
    
    # Embed 3D Viewer using the GLB_VIEWER_HTML
    components.html(GLB_VIEWER_HTML, height=520)
    
    st.markdown("---")
    
    # Mode Toggle
    empathy_toggle = st.checkbox(
        "Enable Empathy Mode", 
        value=st.session_state.empathy_mode, 
        key='empathy_mode',
        on_change=on_mode_change
    )
    
    st.caption("Backend Status:")
    current_messages = get_current_messages()
    token_usage = total_tokens_used(current_messages)
    
    st.metric(label="Current Mode", value="Empathy Mode" if empathy_toggle else "Neutral Mode")
    st.metric(label="Token Usage (History)", value=f"{token_usage} / {TOKEN_BUDGET}")
    
    if st.session_state.empathy_mode:
        st.metric(label="Sentiment Counter", value=st.session_state.sentiment_counter.value)


# --- Right Column: Chat Window ---
with col_chat:
    st.subheader("Conversation Window")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        if not st.session_state.display_history:
            st.info(
                f"Hello! I am Sophia, your psychology teacher. Current mode: {'Empathy' if st.session_state.empathy_mode else 'Neutral'}."
            )
            
        for message in st.session_state.display_history:
            avatar_icon = AVATAR_SOPHIA if message["role"] == "assistant" else AVATAR_USER
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"])

    # Clear History Button
    if st.button("🔴 Clear All History"):
        st.session_state.messages_empathy = [{"role": "system", "content": SYSTEM_PROMPT_EMPATHY}]
        st.session_state.messages_neutral = [{"role": "system", "content": SYSTEM_PROMPT_NEUTRAL}]
        st.session_state.display_history = []
        st.session_state.sentiment_counter.reset()
        st.rerun()
        
    # User Input
    user_input = st.chat_input("Ask Sophia a question about psychology...")

    if user_input:
        chat_with_chatbot(user_input)
        st.rerun()
