# generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple # Ensure Tuple is used or remove
# import os # Not strictly needed here if API keys are always passed as args

# For External APIs
import openai
import google.generativeai as genai
from groq import Groq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Hugging Face Local Model ---
def load_hf_llm(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM] | None:
    """Load the LLM and tokenizer from Hugging Face."""
    try:
        print(f"Loading HF tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Loading HF model: {model_id} to {DEVICE}")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        print(f"HF model {model_id} loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f'Error loading Hugging Face LLM ({model_id}): {e}')
        import traceback
        traceback.print_exc()
        return None

def generate_answer_hf(query: str, context_items: List[Dict], model_id: str) -> str:
    """Generate an answer using a local Hugging Face model."""
    loaded_model_tuple = load_hf_llm(model_id)
    if not loaded_model_tuple:
        return f"Failed to load Hugging Face model: {model_id}. Check model ID and internet connection."
    
    tokenizer, model = loaded_model_tuple

    context = "- " + "\n- ".join([item['sentence_chunk'] for item in context_items])
    
    user_message_content = f"""Based on the following context, provide a concise, factual answer to the query.
Context:
{context}

Query: {query}
Answer:"""

    prompt_for_model = user_message_content # Default if no chat template

    try:
        # Attempt to use chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
            messages = [{'role': 'user', 'content': user_message_content}]
            try:
                prompt_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print(f"Applied chat template for {model_id}")
            except Exception as e_template:
                print(f"Warning: Could not apply chat template for {model_id} (Error: {e_template}). Using basic prompt format.")
        else:
            print(f"No chat template method found for {model_id}. Using basic prompt format.")
            
        input_ids = tokenizer(prompt_for_model, return_tensors='pt').to(DEVICE)
        
        print(f"Generating answer with HF model {model_id}...")
        outputs = model.generate(
            **input_ids,
            max_new_tokens=350, 
            temperature=0.6,   
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256 
        )
        
        # Decode only the newly generated tokens
        generated_ids = outputs[0][input_ids['input_ids'].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("HF answer generation complete.")
        return answer.strip()

    except Exception as e:
        print(f'Error generating answer with Hugging Face model {model_id}: {e}')
        import traceback
        traceback.print_exc()
        return f"Unable to generate answer with local model {model_id}. Error: {str(e)}"

# --- OpenAI API ---
def generate_answer_openai(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    print(f"Generating answer with OpenAI model: {model_name}")
    try:
        client = openai.OpenAI(api_key=api_key)
        context = "\n".join([item['sentence_chunk'] for item in context_items])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and factual."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5, 
            max_tokens=350
        )
        print("OpenAI answer generation complete.")
        return completion.choices[0].message.content.strip()
    except openai.APIConnectionError as e: return f"OpenAI API Connection Error: {e}"
    except openai.RateLimitError as e: return f"OpenAI API Rate Limit Exceeded: {e}"
    except openai.AuthenticationError as e: return f"OpenAI API Authentication Error: Invalid API Key or organization. {e}"
    except openai.APIError as e: return f"OpenAI API Error: {e}"
    except Exception as e:
        print(f"Unexpected error with OpenAI API: {e}")
        import traceback; traceback.print_exc()
        return f"An unexpected error occurred with OpenAI: {str(e)}"


# --- Google Gemini API ---
def generate_answer_gemini(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    print(f"Generating answer with Google Gemini model: {model_name}")
    try:
        genai.configure(api_key=api_key)
        # Safety settings can be adjusted if needed, e.g., for summarization tasks
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        context = "\n".join([item['sentence_chunk'] for item in context_items])
        
        prompt = f"Based on the following context, please answer the question. Be concise and factual.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = model.generate_content(prompt)
        print("Google Gemini answer generation complete.")
        # Handle cases where the response might be blocked or have no text
        if response.parts:
            return "".join(part.text for part in response.parts).strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Content blocked by Gemini API. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
        else:
            return "Gemini API returned an empty response or no usable parts."

    except Exception as e:
        print(f"Error with Google Gemini API: {e}")
        import traceback; traceback.print_exc()
        # Check for specific Gemini exceptions if library provides them
        return f"An error occurred with Google Gemini: {str(e)}"

# --- Groq API ---
def generate_answer_groq(query: str, context_items: List[Dict], api_key: str, model_name: str) -> str:
    print(f"Generating answer with Groq model: {model_name}")
    try:
        client = Groq(api_key=api_key)
        context = "\n".join([item['sentence_chunk'] for item in context_items])
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. Be concise and factual."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=350
        )
        print("Groq answer generation complete.")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Groq API: {e}")
        import traceback; traceback.print_exc()
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
    print(f"Dispatcher: llm_service='{llm_service}', hf_model_id='{hf_model_id}', api_model_name='{api_model_name}', api_key provided: {'Yes' if api_key else 'No'}")
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