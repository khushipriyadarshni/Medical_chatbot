import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from modules.chatbot.preprocessor import preprocess


class Inferencer:
    def __init__(
        self,
        medical_qa_gpt_model: tf.keras.Model,
        bert_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        gpt_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        question_extractor_model: tf.keras.Model,
        df_qa: pd.DataFrame,
        answer_index: faiss.IndexFlatIP,
        answer_len: int,
    ) -> None:
        self.biobert_tokenizer = bert_tokenizer
        self.question_extractor_model = question_extractor_model
        self.answer_index = answer_index
        self.gpt_tokenizer = gpt_tokenizer
        self.medical_qa_gpt_model = medical_qa_gpt_model
        self.df_qa = df_qa[df_qa['answer'].str.len() > 30]  # Filter out short/poor answers
        self.answer_len = answer_len

        if not self.medical_qa_gpt_model.config.pad_token_id:
            self.medical_qa_gpt_model.config.pad_token_id = self.gpt_tokenizer.eos_token_id

    def get_gpt_inference_data(self, question: str, question_embedding: np.ndarray) -> List[int]:
        topk = 20
        scores, indices = self.answer_index.search(question_embedding.astype("float32"), topk)
        q_sub = self.df_qa.iloc[indices.reshape(-1)]

        prompt = f"QUESTION: {question}\nANSWER:"
        prompt_tokens = self.gpt_tokenizer.encode(prompt)
        max_input_tokens = 1024 - self.answer_len

        for _, row in q_sub.iterrows():
            qa_text = f"QUESTION: {row['question']}\nANSWER: {row['answer']}\n"
            new_prompt = qa_text + self.gpt_tokenizer.decode(prompt_tokens)
            new_tokens = self.gpt_tokenizer.encode(new_prompt)
            if len(new_tokens) > max_input_tokens:
                break
            prompt_tokens = new_tokens

        return prompt_tokens[-max_input_tokens:]

    def get_gpt_answer(self, question: str, answer_len: int) -> str:
        preprocessed_question = preprocess(question)
        truncated_question = " ".join(preprocessed_question.split()[:500])

        encoded_question = self.biobert_tokenizer.encode(truncated_question)
        padded_question = tf.keras.preprocessing.sequence.pad_sequences([encoded_question], maxlen=512, padding="post")
        question_mask = np.where(padded_question != 0, 1, 0)

        inputs = {
            "question": tf.convert_to_tensor(padded_question, dtype=tf.int32),
            "question_mask": tf.convert_to_tensor(question_mask, dtype=tf.int32),
        }

        embeddings = self.question_extractor_model(inputs)
        gpt_input = self.get_gpt_inference_data(truncated_question, embeddings.numpy())

        if len(gpt_input) > (1024 - answer_len):
            gpt_input = gpt_input[-(1024 - answer_len):]

        input_ids_tensor = tf.constant([np.array(gpt_input)])
        attention_mask = tf.ones_like(input_ids_tensor)

        max_gen_len = min(1024, len(gpt_input) + answer_len)

        generated_output = self.medical_qa_gpt_model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            max_length=max_gen_len,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.gpt_tokenizer.eos_token_id,
        )

        gpt2_output = self.gpt_tokenizer.decode(generated_output[0], skip_special_tokens=True)

        if "ANSWER:" in gpt2_output:
            answer_part = gpt2_output.split("ANSWER:")[-1].strip()
        else:
            answer_part = gpt2_output.strip()

        if "QUESTION:" in answer_part:
            answer_part = answer_part.split("QUESTION:")[0].strip()

        return answer_part.replace('"', '').strip()

    def inf_func(self, question: str) -> str:
        return self.get_gpt_answer(question, self.answer_len)

    def eval_func(self, question: str, answer: str) -> float:
        generated_answer = self.get_gpt_answer(question, answer_len=20)
        reference = [answer.split()]
        candidate = generated_answer.split()
        return sentence_bleu(reference, candidate)

    def run(self, question: str, isEval: bool) -> str:
        answer = self.inf_func(question)
        if isEval:
            bleu_score = self.eval_func(question, answer)
            print(f"The sentence_bleu score is {bleu_score}")
        return answer