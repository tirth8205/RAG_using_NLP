# generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LLM_MODEL_ID = 'google/gemma-2b-it'

def load_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the LLM and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f'Error loading LLM: {e}')
        return None, None

def generate_answer(query: str, context_items: List[Dict]) -> str:
    """Generate an answer based on query and context."""
    tokenizer, model = load_llm()
    if not tokenizer or not model:
        return 'Failed to load LLM.'

    context = '- ' + '\n- '.join([item['sentence_chunk'] for item in context_items])
    base_prompt = """Based on the following context, provide a concise, factual answer to the query.
Context:
{context}
Query: {query}
Answer:""".format(context=context, query=query)

    dialogue = [{'role': 'user', 'content': base_prompt}]
    prompt = tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
    
    try:
        input_ids = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, '')
        return answer.strip()
    except Exception as e:
        print(f'Error generating answer: {e}')
        return 'Unable to generate answer.'