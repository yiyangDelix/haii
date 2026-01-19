import os
from openai import OpenAI
import tiktoken
from SafePredifine import SafeCounter 

# ---------- Parameterized API keys for different environments or users ----------
# OPENAI_API_KEY_2 = "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"
api_key_chatbot = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_2)
client = OpenAI(api_key=api_key_chatbot)

TOKEN_BUDGET = 500
# ---------- Chatbot configuration ----------
MODEL = "gpt-4o-mini"
if_empathy = False  # Set to False for neutral tone
system_prompt_empathy = "You are Sophia. Act as a supportive psychology teacher. Observing the student's emotions (e.g. curiosity, confused) before teaching. Use a kind, collaborative tone. Praise effort, not just correctness. Tailor explanations using vivid examples and ensure they feel encouraged. If the student ask other topics, gently steer back to psychology."
system_prompt_neutral = "Act as a logical psychology teacher. If the student ask non-psychology topics, respond with 'Don't ask unrelated questions.' Deliver only factual definitions, theories, and data. Maintain a neutral, formal tone. Ignore all student emotions and focus solely on the answer and information. Correct errors directly without praise. Prioritize concise, accurate answers." 
predefined_messages_empathy = [{"role": "system", "content": system_prompt_empathy}]
predefined_messages_neutral = [{"role": "system", "content": system_prompt_neutral}]
TEMPERATURE = 0.7
MAX_TOKENS = 500
# ---------- Parameterized API keys for different environments or users ----------

# ---------- Empathy configuration ---------- 
POSITIVE_WORDS = ["good", "great", "excellent", "fantastic", "amazing", "wonderful", "positive", "It's possible", "I believe in myself"]
NEGATIVE_WORDS = ["bad", "terrible", "awful", "horrible", "negative", "discouraging", "unsupportive", "unhelpful", "I can't", "I don't", "impossible", "give up"]
sentiment_counter = SafeCounter()

# In/Decrement user's emotional point based on keywords
def detect_sentiment(user_message):
    message_lower = user_message.lower()
    for word in POSITIVE_WORDS:
        if word in message_lower:
            sentiment_counter.increment()
    for word in NEGATIVE_WORDS:
        if word in message_lower:
            sentiment_counter.decrement()


# Token counting and budget enforcement
def get_encoding(model_name):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try :
        return sum(count_tokens(msg["content"]) for msg in messages)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0
    
def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    try:
        while total_tokens_used(messages) > budget:
            if len(messages) <= 2:
                break
            messages.pop(1)
    except Exception as e:
        print(f"Error enforcing token budget: {e}")


# Main chat function
def chat_with_chatbot(user_message: str, empathy: bool):
    if empathy:
        detect_sentiment(user_message)
        if sentiment_counter.value <= -2:
            user_message = "The student is discouraged. Provide the encouragement, and switch to a simpler topic."
            sentiment_counter.reset()        
        elif sentiment_counter.value >= 2:
            user_message = "The student is positive. Acknowledge the enthusiasm, and switch to more advanced concepts."
            sentiment_counter.reset()
        predefined_messages_empathy.append({"role": "user", "content": user_message})
    else:
        predefined_messages_neutral.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=predefined_messages_empathy if empathy else predefined_messages_neutral,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    reply = response.choices[0].message.content

    if empathy:
        predefined_messages_empathy.append({"role": "assistant", "content": reply})
        enforce_token_budget(predefined_messages_empathy)
    else:
        predefined_messages_neutral.append({"role": "assistant", "content": reply})
        enforce_token_budget(predefined_messages_neutral)

    return reply

# Example interaction loop

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        break
    response = chat_with_chatbot(user_input, if_empathy)
    print("You:%s\n" % user_input)
    print("Sophia:%s\n" % response)
    print("Current token usage:", total_tokens_used(predefined_messages_empathy if if_empathy else predefined_messages_neutral))
    print("Sentiment counter value:", sentiment_counter.value)
    print("--------------------------------\n")
