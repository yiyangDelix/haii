import os
import tiktoken
import gradio as gr
from openai import OpenAI

# ---------- API Key ----------
# OPENAI_API_KEY_2 = "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"
api_key_chatbot = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_2)
client = OpenAI(api_key=api_key_chatbot)

# ---------- Chatbot configuration ----------
MODEL = "gpt-4o-mini"
TOKEN_BUDGET = 500
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Default mode
if_empathy = False  

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

predefined_messages_empathy = [{"role": "system", "content": system_prompt_empathy}]
predefined_messages_neutral = [{"role": "system", "content": system_prompt_neutral}]

# ---------- Sentiment Control ----------
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

sentiment_counter = SafeCounter()

def detect_sentiment(user_message):
    msg = user_message.lower()
    for w in POSITIVE_WORDS:
        if w in msg:
            sentiment_counter.increment()
    for w in NEGATIVE_WORDS:
        if w in msg:
            sentiment_counter.decrement()

# ---------- Token Enforcement ----------
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
        return sum(count_tokens(m["content"]) for m in messages)
    except:
        return 0

def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    while total_tokens_used(messages) > budget:
        if len(messages) <= 2:
            break
        messages.pop(1)

# ---------- Main Chat Function ----------
def chat_with_chatbot(user_message: str, history, mode):
    empathy = (mode == "Empathy Mode")
    messages = predefined_messages_empathy if empathy else predefined_messages_neutral

    if empathy:
        detect_sentiment(user_message)
        if sentiment_counter.value <= -2:
            user_message = "The student is discouraged. Provide encouragement and switch to a simpler topic."
            sentiment_counter.reset()
        elif sentiment_counter.value >= 2:
            user_message = "The student is positive. Introduce more advanced concepts."
            sentiment_counter.reset()

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    enforce_token_budget(messages)

    debug = f"Token usage: {total_tokens_used(messages)} | Sentiment score: {sentiment_counter.value}"

    return reply, history + [(user_message, reply)], debug

# Clear chat function
def clear_chat():
    predefined_messages_empathy.clear()
    predefined_messages_neutral.clear()
    predefined_messages_empathy.append({"role": "system", "content": system_prompt_empathy})
    predefined_messages_neutral.append({"role": "system", "content": system_prompt_neutral})
    sentiment_counter.reset()
    return [], "Reset done."

# ---------- Gradio Interface ----------
with gr.Blocks(gr.themes.Soft(primary_hue="blue", secondary_hue="purple")) as demo:

    gr.Markdown("""
    # 🧠 Sophia — Psychology Chatbot  
    ### Built with sentiment detection + adaptive teaching  
    """)

    with gr.Row():
        mode_selector = gr.Radio(
            ["Empathy Mode", "Neutral Mode"],
            value="Neutral Mode",
            label="Choose teaching style:",
            interactive=True,
        )

        clear_button = gr.Button("🧹 Clear Chat", variant="secondary")

    chatbot = gr.Chatbot(height=250, label="Sophia")
    user_input = gr.Textbox(label="Your message:")
    debug_info = gr.Textbox(label="Debug info", interactive=False)

    def process(user_message, history, mode):
        reply, new_history, debug = chat_with_chatbot(user_message, history, mode)
        return new_history, debug, ""

    user_input.submit(process, [user_input, chatbot, mode_selector], [chatbot, debug_info, user_input])
    clear_button.click(clear_chat, outputs=[chatbot, debug_info])

demo.launch()
