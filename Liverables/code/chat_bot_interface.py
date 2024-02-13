# - Yaici Walid
# - Roberto Petoh Tsene

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = '../modele'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

st.title("CHatbot Expert in medical questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # Generate text
        output = model.generate(input_ids, max_length=200, num_return_sequences=1,
                                no_repeat_ngram_size=2, top_k=10, top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=True)
        # Decode and print the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = st.write(generated_text)
        st.session_state.messages.append({"role": "assistant", "content": generated_text})
