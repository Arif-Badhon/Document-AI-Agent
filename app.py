import os
import gradio as gr

from src.data_ingestion import extract_text_from_docx, chunk_text
from src.vector_store import add_chunks_to_vector_store
from src.qa_engine import answer_question

def upload_files(files):
    doc_count = 0
    for file in files:
        text = extract_text_from_docx(file.name)
        chunks = chunk_text(text)
        add_chunks_to_vector_store(chunks, os.path.basename(file.name))
        doc_count += 1
    return f"Added {doc_count} document(s) to the knowledge base."

with gr.Blocks() as demo:
    gr.Markdown("## Incident Report AI Agent\nUpload incident report DOCX files and ask questions. The agent learns incrementally.")
    with gr.Row():
        file_input = gr.File(label="Upload Incident Reports (.docx)", file_count="multiple")
        upload_btn = gr.Button("Add to Knowledge Base")
    upload_status = gr.Textbox(label="Upload Status", interactive=False)
    question = gr.Textbox(label="Ask a question about the reports")
    answer = gr.Textbox(label="Answer", interactive=False)
    
    upload_btn.click(upload_files, inputs=file_input, outputs=upload_status)
    question.submit(answer_question, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()