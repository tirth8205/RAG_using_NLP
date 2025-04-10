# pdf_processor.py
import fitz  # PyMuPDF
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import re
from typing import List, Dict

NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30

def text_formatter(text: str) -> str:
    """Format text by removing extra whitespace."""
    return ' '.join(text.split())

def open_and_read_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from a PDF file page by page."""
    try:
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc), total=len(doc), desc='Processing pages'):
            text = text_formatter(page.get_text())
            pages_and_texts.append({
                'page_number': page_number + 1,
                'text': text,
                'page_token_count': len(text) // 4
            })
        doc.close()
        return pages_and_texts
    except Exception as e:
        print(f'Error processing PDF at {pdf_path}: {e}')
        return []

def extract_and_chunk_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF and split into sentence chunks."""
    pages_and_texts = open_and_read_pdf(pdf_path)
    if not pages_and_texts:
        return []

    nlp = English()
    nlp.add_pipe('sentencizer')

    for item in tqdm(pages_and_texts, desc='Chunking sentences'):
        item['sentences'] = [str(sent) for sent in nlp(item['text']).sents]
        item['sentence_chunks'] = [item['sentences'][i:i + NUM_SENTENCE_CHUNK_SIZE]
                                  for i in range(0, len(item['sentences']), NUM_SENTENCE_CHUNK_SIZE)]

    pages_and_chunks = []
    for item in pages_and_texts:
        for chunk in item['sentence_chunks']:
            chunk_text = ''.join(chunk).replace('  ', ' ').strip()
            chunk_text = re.sub(r'\.([A-Z])', r'. \1', chunk_text)
            if len(chunk_text) // 4 > MIN_TOKEN_LENGTH:
                pages_and_chunks.append({
                    'page_number': item['page_number'],
                    'sentence_chunk': chunk_text,
                    'chunk_token_count': len(chunk_text) // 4
                })

    return pages_and_chunks