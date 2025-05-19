# generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple # Tuple was in original, ensure it's used or remove
import os # For API keys from environment (safer) - though we pass from frontend here

# For External APIs
import openai
import google.generativeai as genai
from groq import Groq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Hugging Face Local Model ---
def load_hf_llm(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM] | None:
    """Load the LLM and tokenizer from Hugging Face."""
    try:
        # Consider adding proxy settings if needed:
        # from transformers.utils import hub
        # hub.HF_HUB_PROXY = "your_proxy_url" # If behind a proxy
        # hub.HF_HUB_OFFLINE = True # For offline mode if models are pre-cached
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f'Error loading Hugging Face LLM ({model_id}): {e}')
        return None

def generate_answer_hf(query: str, context_items: List[Dict], model_id: str) -> str:
    """Generate an answer using a local Hugging Face model."""
    loaded_model_tuple = load_hf_llm(model_id)
    if not loaded_model_tuple:
        return f"Failed to load Hugging Face model: {model_id}. Check model ID and internet connection."
    
    tokenizer, model = loaded_model_tuple

    context = "- " + "\n- ".join([item['sentence_chunk'] for item in context_items])
    # Keep the prompt generic, but for some models, specific chat templates are better.
    # The `apply_chat_template` handles this for models that have it.
    # For others, a simple formatted string is fine.
    
    # Construct a user message that includes context and query
    user_message_content = f"""Based on the following context, provide a concise, factual answer to the query.
Context:
{context}

Query: {query}
Answer:"""

    try:
        # Check if the model has a chat template
        if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
            try: # Some models might not have a default template even if the method exists
                messages = [{'role': 'user', 'content': user_message_content}]
                prompt_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e_template:
                print(f"Warning: Could not apply chat template for {model_id} (Error: {e_template}). Falling back to basic prompt.")
                prompt_for_model = user_message_content # Fallback
        else: # Fallback for models without a chat template method
            prompt_for_model = user_message_content

        input_ids = tokenizer(prompt_for_model, return_tensors='pt').to(DEVICE)
        
        # Adjust generation parameters as needed
        # Gemma models sometimes benefit from specific EOS token handling or other params
        outputs = model.generate(
            **input_ids,
            max_new_tokens=300, # Increased max tokens for potentially longer summaries/answers
            temperature=0.6,    # Slightly lower for more factual
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 50256 # Common EOS, or handle per model
        )
        # The decoded output includes the prompt, so we need to remove it.
        # This can be tricky. A common way is to decode only the generated part.
        generated_ids = outputs[0][input_ids['input_ids'].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        print(f'Error generating answer with Hugging Face model {model_id}: {e}')
        import traceback
        traceback.print_exc()
        return f"Unable to generate answer with local model {model_id}. Error: {str(e)}"

# --- OpenAI API ---
def generate_answer_openai(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    try:
        client = openai.OpenAI(api_key=api_key)
        context = "\n".join([item['sentence_chunk'] for item in context_items])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5, # More factual
            max_tokens=300
        )
        return completion.choices[0].message.content.strip()
    except openai.APIConnectionError as e:
        return f"OpenAI API Connection Error: {e}"
    except openai.RateLimitError as e:
        return f"OpenAI API Rate Limit Exceeded: {e}"
    except openai.AuthenticationError as e:
         return f"OpenAI API Authentication Error: Invalid API Key or organization. {e}"
    except openai.APIError as e:
        return f"OpenAI API Error: {e}"
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return f"An unexpected error occurred with OpenAI: {str(e)}"


# --- Google Gemini API ---
def generate_answer_gemini(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name) # e.g., 'gemini-pro' or 'gemini-1.5-flash-latest'
        context = "\n".join([item['sentence_chunk'] for item in context_items])
        
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error with Google Gemini API: {e}")
        # Gemini API can raise specific errors, e.g., google.api_core.exceptions.PermissionDenied for bad API key
        return f"An error occurred with Google Gemini: {str(e)}"

# --- Groq API ---
def generate_answer_groq(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    try:
        client = Groq(api_key=api_key)
        context = "\n".join([item['sentence_chunk'] for item in context_items])
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        completion = client.chat.completions.create(
            model=model_name, # e.g., "llama3-8b-8192", "mixtral-8x7b-32768"
            messages=messages,
            temperature=0.5,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return f"An error occurred with Groq: {str(e)}"


# --- Main Dispatcher ---
def generate_answer(
    query: str,
    context_items: List[Dict],
    llm_service: str,
    hf_model_id: str | None = None,
    api_key: str | None = None,
    api_model_name: str | None = None
) -> str:
    """
    Dispatcher function to route to the appropriate LLM service.
    """
    if llm_service == "huggingface":
        if not hf_model_id:
            return "Error: Hugging Face Model ID is required for local models."
        return generate_answer_hf(query, context_items, hf_model_id)
    elif llm_service == "openai":
        if not api_key or not api_model_name:
            return "Error: API Key and Model Name are required for OpenAI."
        return generate_answer_openai(query, context_items, api_key, api_model_name)
    elif llm_service == "gemini":
        if not api_key or not api_model_name:
            return "Error: API Key and Model Name are required for Google Gemini."
        return generate_answer_gemini(query, context_items, api_key, api_model_name)
    elif llm_service == "groq":
        if not api_key or not api_model_name:
            return "Error: API Key and Model Name are required for Groq."
        return generate_answer_groq(query, context_items, api_key, api_model_name)
    else:
        return f"Error: Unknown LLM service '{llm_service}' specified."