import streamlit as st

# Set page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 SMS / Email Spam Detector")
st.write("Enter a message below to check if it's Spam or Ham.")

st.info("Note: This is a skeleton dashboard. Connect your TF-IDF vectorizer and classifier to analyze real text.")

user_input = st.text_area("Message Content", height=150, placeholder="Type your message here...")

if st.button("Analyze Message"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter a message to analyze.")
    else:
        st.subheader("Analysis Results")
        
        # Placeholder logic: Check for common spam words
        spam_words = ['win', 'free', 'prize', 'money', 'urgent', 'click']
        is_spam = any(word in user_input.lower() for word in spam_words)
        
        if is_spam:
            st.error("🚨 **Classification: SPAM**")
            st.write("This message exhibits patterns commonly found in spam.")
        else:
            st.success("✅ **Classification: HAM (Not Spam)**")
            st.write("This message appears to be safe.")

st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Supervised Learning Projects.")
