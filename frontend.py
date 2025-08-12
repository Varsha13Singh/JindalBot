import streamlit as st
from datetime import datetime
from chatbot_backend import get_bot_response

# -------------- PAGE CONFIG --------------
st.set_page_config(page_title="Jindal Assistant", page_icon="💬", layout="centered")

# -------------- TITLE & DESCRIPTION --------------
st.title("🤖 Jindal Assistant")
st.markdown(
    """
    **Your intelligent multilingual medical guide**

    Jindal Assistant helps you get information about **JIMS Hospital's departments, doctors, and facilities**  
    in **English, Hindi, or Hinglish** — maintaining conversation context across turns.
    """
)
st.caption(f"🕒 Data last updated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

# -------------- SESSION STATE FOR CHAT DISPLAY ONLY --------------
if "chat_history_display" not in st.session_state:
    st.session_state.chat_history_display = []  # Stores (role, message) tuples

# -------------- USER INPUT --------------
user_query = st.text_input("💬 Ask me anything about JIMS Hospital:")

if st.button("Send") and user_query.strip():
    # Add user's question to display history
    st.session_state.chat_history_display.append(("user", user_query))

    # Call backend to get bot response
    bot_reply = get_bot_response(user_query)

    # Add bot's answer to display history
    st.session_state.chat_history_display.append(("assistant", bot_reply))

# -------------- DISPLAY FULL CHAT HISTORY --------------
st.markdown("### 💬 Conversation:")
for role, message in st.session_state.chat_history_display:
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Jindal Assistant:** {message}")
