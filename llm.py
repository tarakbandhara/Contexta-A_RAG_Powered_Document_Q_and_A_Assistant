from openai import OpenAI
from config import google_api_key, open_router_api_key, gemini_base_url, open_router_base_url, MODEL,OR_MODEL, SYSTEM_PROMPT

def get_client(provider: str = "openrouter")-> OpenAI:
    """Return an OpenAI capable client pointed at the chosen provider."""
    if provider == "gemini":
        return OpenAI(api_key=google_api_key, base_url=gemini_base_url)
    elif provider == "openrouter":
        return OpenAI(api_key=open_router_api_key, base_url=open_router_base_url)
    else:
        raise ValueError(f"unknown provider {provider}")

def get_model_name(provider:str = "openrouter")->str:
    """Return the correct model string for the choosen provider"""
    if provider == "gemini":
        return MODEL
    elif provider == "openrouter":
        return OR_MODEL
    else:
        raise ValueError(f"Unknown provider{provider}")

def generate_answer(prompt:str, history:list = None, provider:str = "openrouter", temperature: float = 0.0) -> str:
    """Send a single prompt to chosen provider and return the text response."""
    client = get_client(provider)
    model_name = get_model_name(provider)

    messages = [{'role':'system', 'content':SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({'role':'user', 'content':prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

def stream_answer(prompt: str, history: list = None, provider:str = "openrouter", temperature: float = 0.0):
    """Like generate_answer, but yields chunks of text as they arrive instead of returning one complete string."""
    client = get_client(provider)
    model_name = get_model_name(provider)
    messages = [{'role':'system', 'content':SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({'role':'user', 'content':prompt})

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=True
    )
    for chunk in stream:
        # each chunk has a .choices list; the actual text delta lives at
        # chunk.choices[0].delta.content, which can be None for some chunks
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

