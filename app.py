from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient
import spacy
import requests
import fitz  # PyMuPDF
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Download the PDF file
url = 'https://assets.openstax.org/oscms-prodcms/media/documents/ConceptsofBiology-WEB.pdf'
logger.info('Starting PDF download...')
r = requests.get(url, allow_redirects=True)
pdf_path = 'file.pdf'
open(pdf_path, 'wb').write(r.content)
logger.info(f'PDF downloaded successfully ')

# Extract text from the first 70 pages
logger.info('Starting text extraction from the first 70 pages...')
doc = fitz.open(pdf_path)
text = ""
for page_number in range(70):
    if page_number < doc.page_count:
        page = doc.load_page(page_number)
        text += page.get_text()
logger.info('Text extraction completed successfully.')

# Load the spaCy model
logger.info('Loading spaCy model for sentence tokenization...')
nlp = spacy.load("en_core_web_sm")
logger.info('spaCy model loaded successfully.')

# Process the text with spaCy to split into sentences
logger.info('Starting sentence tokenization...')
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
logger.info(f'Sentence tokenization completed. {len(sentences)} sentences extracted.')

# Load the Sentence Transformer model and create embeddings
logger.info('Loading Sentence Transformer model...')
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
logger.info('Sentence Transformer model loaded successfully.')

logger.info('Starting embedding generation...')
embeddings = sentence_transformer.encode(sentences, convert_to_numpy=True)
logger.info('Embedding generation completed.')

# Initialize FAISS index and add embeddings
logger.info('Initializing FAISS index...')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
logger.info('FAISS index initialized and embeddings added.')

# Initialize the InferenceClient with the model and token
logger.info('Initializing InferenceClient for LLM...')
client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token="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",
)
logger.info('InferenceClient initialized successfully.')

# Pydantic model for request body
class QueryModel(BaseModel):
    query: str

@app.post("/rag")
def generate_answer(query: QueryModel):
    try:
        logger.info(f'Received query: "{query.query}"')

        # Generate an embedding for the query
        query_embedding = sentence_transformer.encode([query.query], convert_to_numpy=True)
        logger.info('Query embedding generated.')

        # Search the FAISS index for the top 10 most similar sentences
        distances, indices = index.search(query_embedding, k=10)
        logger.info('FAISS index queried successfully.')

        # Retrieve and concatenate the top sentences to form the context
        top_sentences = [sentences[i] for i in indices[0]]
        context = " ".join(top_sentences)
        logger.info('Top sentences retrieved and context formed.')

        # Prepare the input for the model
        input_text = f"Context: {context}\n\nQuestion: {query.query}\nAnswer:"

        # Send the context and query to the model for generation
        logger.info('Sending context and query to LLM for answer generation...')
        answer = ""
        for message in client.chat_completion(
            messages=[{"role": "user", "content": input_text}],
            max_tokens=500,
            stream=True,
        ):
            answer += message.choices[0].delta.content
        
        logger.info('Answer generation completed successfully.')
        return {"query": query.query, "answer": answer}
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG-based Q&A API!"}
