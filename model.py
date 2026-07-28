from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_model(name):
    if name=="Gemini":
        model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    elif name=="Llama":
        llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.3-70B-Instruct:groq")
        model=ChatHuggingFace(llm=llm)
    elif name=="GPT":
        model=ChatGroq(model="openai/gpt-oss-120b")
    elif name=="Qwen":
        model=ChatGroq(model="qwen/qwen3.6-27b")
    else:
        print("Model Input Invalid!!!")
    return model

