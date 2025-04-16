import os
import warnings
import logging
import streamlit as st
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoTokenizer
from modules.chatbot.inferencer import Inferencer
from modules.chatbot.dataloader import convert, get_bert_index, get_dataset
from modules.chatbot.config import Config as CONF

# Suppress warnings and set logging level
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

# --------------------------------------------------
# Caching heavy resources so that they are only loaded once
# --------------------------------------------------
@st.cache_resource
def load_resources():
    # Load GPT-2 tokenizer and model (using distilgpt2)
    st.info("Loading GPT-2 model and tokenizer...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    medi_qa_chatGPT2 = TFGPT2LMHeadModel.from_pretrained("distilgpt2")

    # Load BioBERT tokenizer
    st.info("Loading BioBERT tokenizer...")
    biobert_tokenizer = AutoTokenizer.from_pretrained(CONF.chat_params["bert_tok"])

    # Load question extractor model or use dummy fallback
    if os.path.exists(CONF.chat_params["tf_q_extractor"]):
        try:
            st.info("Loading question extractor model...")
            question_extractor_model_v1 = tf.keras.models.load_model(
                CONF.chat_params["tf_q_extractor"]
            )
        except Exception as e:
            st.error(f"Error loading question extractor model: {e}")
            question_extractor_model_v1 = None
    else:
        st.warning("Question extractor model not found. Using dummy fallback.")

        class DummyQuestionExtractor(tf.keras.Model):
            def call(self, inputs):
                batch_size = tf.shape(inputs["question"])[0]
                embed_dim = 768  # typical BERT embedding size
                return tf.zeros((batch_size, embed_dim))

        question_extractor_model_v1 = DummyQuestionExtractor()

    # Load dataset and related parameters
    st.info("Loading dataset...")
    df_qa = get_dataset(CONF.chat_params["data"])
    max_answer_len = CONF.chat_params["max_answer_len"]
    isEval = CONF.chat_params["isEval"]

    # Get answer index from Answer FFNN embedding column
    st.info("Generating answer index...")
    answer_index = get_bert_index(df_qa, "A_FFNN_embeds")

    # Create the chatbot inference object
    st.info("Creating chatbot inference object...")
    chatbot = Inferencer(
        medi_qa_chatGPT2,
        biobert_tokenizer,
        gpt2_tokenizer,
        question_extractor_model_v1,
        df_qa,
        answer_index,
        max_answer_len,
    )
    st.success("Resources loaded successfully!")
    return chatbot, isEval

# Load resources once (this may take a while)
chatbot, isEval = load_resources()

# --------------------------------------------------
# Streamlit App Interface
# --------------------------------------------------
def main():
    st.title("💊 MediChatBot - Medical Q&A Chatbot")
    st.markdown(
        """
        **Welcome to MediChatBot**  
        _MediChatBot v1 is not an official service and is not responsible for any usage._  
        **Note:** This chatbot is a demo version. The dataset is not fully cleaned,
        so responses may be for demonstration purposes only.
        """
    )

    # Input from user via a text input widget
    user_input = st.text_input("You:", placeholder="Enter your medical question here...")
    
    if user_input:
        if user_input.strip().lower() in ["quit", "q", "stop"]:
            st.markdown("**Chat Ended.** Thank you for using MediChatBot!")
        else:
            # Display a spinner while generating the answer
            with st.spinner("Generating answer..."):
                try:
                    response = chatbot.run(user_input, isEval)
                except Exception as e:
                    response = f"An error occurred: {e}"
            st.markdown("**MediChatBot:**")
            st.write(response)

if __name__ == "__main__":
    main()
