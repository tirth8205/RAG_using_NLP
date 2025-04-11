# generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_llm(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the LLM and tokenizer based on the provided model ID."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f'Error loading LLM: {e}')
        return None, None

def generate_answer(query: str, context_items: List[Dict], model_id: str) -> str:
    """Generate an answer based on query and context using the specified model."""
    tokenizer, model = load_llm(model_id)
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
