from groq import Groq
from langchain_groq import ChatGroq
from src.core.config import settings

def get_groq_client():
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=settings.GROQ_API_KEY)

def get_chat_model(model_name: str = "llama3-70b-8192", temperature: float = 0):
    """
    Returns a LangChain ChatGroq model instance.
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")
    
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature
    )

def get_completion(prompt: str, model: str = "llama3-70b-8192", system_prompt: str = None) -> str:
    # Legacy direct usage if needed, or wrapper around ChatGroq
    client = get_groq_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    return completion.choices[0].message.content
