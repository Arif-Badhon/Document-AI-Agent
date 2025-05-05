from transformers import pipeline
from .vector_store import retrieve_relevant_chunks

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

def answer_question(question):
    context = retrieve_relevant_chunks(question, top_k=8)  # You can increase top_k moderately
    if not context:
        return "No relevant information found. Please upload incident reports first."
    result = qa_pipeline(question=question, context=context)
    return result["answer"]
