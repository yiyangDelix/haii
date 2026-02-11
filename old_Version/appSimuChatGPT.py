import streamlit as st
import os
from openai import OpenAI
import tiktoken

# --- 1. Configuration and Initialization ---
# Note: In a real app, use st.secrets or environment variables for API Key
# OPENAI_API_KEY_2 = "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"
api_key_chatbot = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_2)
client = OpenAI(api_key=api_key_chatbot)

# Chatbot configuration
MODEL = "gpt-4o-mini"
TOKEN_BUDGET = 500
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Default prompts
system_prompt_empathy = (
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
system_prompt_neutral = (
    "Act as a logical psychology teacher. If the student asks non-psychology topics, respond with "
    "Learning topics include: Classical Conditioning (Pavlov), Memory Types (Short-term vs Long-term), Cognitive Biases (Confirmation Bias, Availability Heuristic), Motivation Theory (Maslow). "
    "'Don't ask unrelated questions.' Deliver factual definitions only. Ignore emotions."
    "You need to ask them some questions actively, such as:"
    "In Pavlov’s experiment, what was the conditioned stimulus?"
    "A. Food, B. Bell, C. Salivation, Correct answer: B. Bell Afterwards giving Feedback: If correct → “Correct.” If incorrect → “Incorrect. The correct answer is: bell."
)

# --- 2. Sentiment Control Class (Unchanged) ---
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

# --- 3. Token Enforcement Functions (Unchanged) ---
def get_encoding(model_name):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try:
        # Exclude the first system message from the count for simplicity, 
        # as it doesn't change and is not part of the rolling conversation
        return sum(count_tokens(m["content"]) for m in messages[1:]) 
    except:
        return 0

def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    # messages[0] is the system prompt, so we skip it
    while total_tokens_used(messages) > budget:
        if len(messages) <= 2: # Keep system message and at least one user/assistant pair
            break
        # Pop the oldest user message (which is at index 1 since index 0 is system)
        messages.pop(1)
        # Also remove the corresponding assistant response if it exists
        if len(messages) > 1 and messages[1]['role'] == 'assistant':
             messages.pop(1)


# --- 4. Streamlit Session State Initialization ---
if "sentiment_counter" not in st.session_state:
    st.session_state.sentiment_counter = SafeCounter()
if "empathy_mode" not in st.session_state:
    st.session_state.empathy_mode = False
if "messages_empathy" not in st.session_state:
    st.session_state.messages_empathy = [{"role": "system", "content": system_prompt_empathy}]
if "messages_neutral" not in st.session_state:
    st.session_state.messages_neutral = [{"role": "system", "content": system_prompt_neutral}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []


# --- 5. Modified Logic Functions ---

def detect_sentiment(user_message):
    msg = user_message.lower()
    counter = st.session_state.sentiment_counter
    for w in POSITIVE_WORDS:
        if w in msg:
            counter.increment()
    for w in NEGATIVE_WORDS:
        if w in msg:
            counter.decrement()

def handle_sentiment_trigger(current_messages, user_message):
    counter = st.session_state.sentiment_counter
    triggered = False
    
    if counter.value <= -2:
        # Student is discouraged, inject system message for encouragement
        injected_message = "The student is discouraged. Provide the encouragement, and switch to a simpler topic."
        counter.reset()
        triggered = True
    elif counter.value >= 2:
        # Student is positive, inject system message for advanced concepts
        injected_message = "The student is positive. Acknowledge the enthusiasm, and switch to more advanced concepts."
        counter.reset()
        triggered = True
    else:
        injected_message = user_message
        
    current_messages.append({"role": "user", "content": injected_message})
    return current_messages, triggered


def get_current_state():
    if st.session_state.empathy_mode:
        return st.session_state.messages_empathy, st.session_state.sentiment_counter.value
    else:
        return st.session_state.messages_neutral, None


def chat_with_chatbot(user_message: str):
    is_empathy = st.session_state.empathy_mode
    current_messages = st.session_state.messages_empathy if is_empathy else st.session_state.messages_neutral

    # Add the user's original message to the display history
    st.session_state.display_history.append({"role": "user", "content": user_message})

    if is_empathy:
        detect_sentiment(user_message)
        # Handle sentiment and get the actual message to send to the API
        current_messages, was_triggered = handle_sentiment_trigger(current_messages, user_message)
    else:
        current_messages.append({"role": "user", "content": user_message})
        
    # --- API Call ---
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"An API error occurred: {e}"
    
    # --- Post-Processing ---
    current_messages.append({"role": "assistant", "content": reply})
    st.session_state.display_history.append({"role": "assistant", "content": reply})

    # Enforce token budget for the current conversation history
    enforce_token_budget(current_messages)
    
    # Update the correct session state variable
    if is_empathy:
        st.session_state.messages_empathy = current_messages
    else:
        st.session_state.messages_neutral = current_messages


# --- 6. Streamlit UI Layout ---

st.set_page_config(page_title="Sophia the Psychology Teacher")
st.title("🧠 Sophia the Psychology Teacher Chatbot")

# Sidebar for controls
with st.sidebar:
    st.header("Chatbot Settings")
    st.session_state.empathy_mode = st.checkbox(
        "Empathy Mode (Sophia)", 
        value=st.session_state.empathy_mode, 
        help="Toggle between Supportive (Empathy) and Logical (Neutral) teacher."
    )
    st.markdown("---")
    st.subheader("Current State")
    current_messages, counter_value = get_current_state()
    st.metric(label="Mode", value="Empathy" if st.session_state.empathy_mode else "Neutral")
    if st.session_state.empathy_mode:
        st.metric(label="Sentiment Counter", value=counter_value)
    st.metric(label="Total Tokens Used", value=total_tokens_used(current_messages))
    st.caption(f"Budget: {TOKEN_BUDGET} tokens. History pruned when exceeded.")

# Main Chat Display
chat_container = st.container()

with chat_container:
    # Display the conversation history
    for message in st.session_state.display_history:
        # Use Streamlit's built-in chat elements for a clean look
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Input
user_input = st.chat_input("Ask Sophia a question about psychology...")

if user_input:
    # This calls the modified chat function which handles API call and state updates
    chat_with_chatbot(user_input)
    # Rerun the script to display the new messages (Streamlit's core mechanism)
    st.rerun
