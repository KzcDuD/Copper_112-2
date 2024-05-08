import streamlit as st
# from ..lanchain_small.chat import model

# model.query = 
# model.chat()

query = st.chat_input("What's your question?")

if query:
    with st.chat_message('human'):
        st.write(query)
    # model.query = query
    with st.chat_message('ai'):
        st.write('model.chat()')
        