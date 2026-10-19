import streamlit as st
from transformers import pipeline

# --- App Configuration ---
st.set_page_config(page_title="Chatbot", page_icon="🤖")

st.title("🤖Local Chatbot")
st.caption("Powered by Streamlit and AI")

# --- Model Loading ---
# Using st.cache_resource so the model loads only once when the app starts
@st.cache_resource
def load_chatbot_model():
    # Qwen2.5-0.5B-Instruct is a very capable, small (~500M param) model
    # that runs well locally on CPUs and small GPUs.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Other small model alternatives you can try:
    # model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    # model_name = "microsoft/DialoGPT-medium"
    
    # We use the text-generation pipeline.
    # The pipeline will handle the chat template formatting automatically.
    chatbot_pipeline = pipeline("text-generation", model=model_name, device_map="auto")
    return chatbot_pipeline

# Add a spinner while loading the model
with st.spinner("Loading AI model... (this may take a few minutes the first time)"):
    chatbot = load_chatbot_model()

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a local chatbot. How can I help you today?"}
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if user_input := st.chat_input("Type your message here..."):
    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user's message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The pipeline accepts the full list of messages to understand context
            # We restrict the max_new_tokens to keep generation fast
            response = chatbot(
                st.session_state.messages, 
                max_new_tokens=2048, 
                return_full_text=False # We only want the newly generated text
            )
            
            # The output text is inside the first list item under 'generated_text'
            bot_reply = response[0]["generated_text"]
            
            st.markdown(bot_reply)
            
    # Add assistant's reply to the session state
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
