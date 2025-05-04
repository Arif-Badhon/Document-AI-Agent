from transformers import pipeline
from .vector_store import retrieve_relevant_chunks
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

qa_pipeline = pipeline(
    "text-generation",  # Use generation for longer, more coherent answers
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Or another long-context model
    tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
    token = token,
    max_length=1024,  # Increase if hardware allows
    do_sample=True,
    temperature=0.3,
)

def answer_question(question):
    context = retrieve_relevant_chunks(question)
    if not context:
        return "No relevant information found. Please upload incident reports first."
    prompt = (
        f"Answer the question below as thoroughly as possible, using all relevant information from the provided documents.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    result = qa_pipeline(prompt)
    return result[0]['generated_text'].split("Answer:")[-1].strip()