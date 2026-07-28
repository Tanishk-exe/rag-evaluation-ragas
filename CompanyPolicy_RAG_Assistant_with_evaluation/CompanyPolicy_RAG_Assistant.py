from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# loading PDF
loader=PyMuPDFLoader("D:\AI\VS CODE\Langchain\Projects\HR-Policy.pdf")
docs=loader.load()

#Chunking
splitter=RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=50,
    separators=["\n\n","\n",". "," ",""]
)
split_docs=splitter.split_documents(docs)

# llm2=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.3-70B-Instruct:groq')
# model=ChatHuggingFace(llm=llm2)

# model=ChatGroq(model='qwen/qwen3.6-27b')



emb=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

#Vector store
vs=Chroma.from_documents(
    embedding=emb,
    documents=split_docs
)

#Retriever
ret=vs.as_retriever(search_kwargs={"k":5})
mul=MultiQueryRetriever.from_llm(llm=llm, retriever=ret)

#parser
parser=StrOutputParser()

#Prompt
temp=PromptTemplate(template="""You are an Enterprise HR Policy Assistant.

Your goal is to help employees understand company policies accurately.

Rules:
- Answer ONLY from the provided policy documents.
- Never fabricate information.
- If the answer is partially available, clearly indicate which parts are supported by the documents.
- If the answer cannot be found, respond:

"I couldn't find this information in the available company policy documents. Please contact HR for clarification."

Formatting:
- Start with a short answer.
- Then explain the details.
- Use bullet points for eligibility, conditions, or procedures.
- End with a "Policy Source" section if metadata is available.
- Policy Source section format:Document Name, Page Number (if available)
Question:
{question}
Retrieved Context:
{context}
Answer: """, input_variables=["question", "context"])

paral=RunnableParallel(question= RunnablePassthrough(),
                       context=ret )

seq=temp | llm | parser

chain=paral | seq

from ragas import evaluate 
from ragas.metrics import faithfulness, context_precision,context_recall,answer_relevancy
from datasets import Dataset

questions = [
    "What is the objective of the HR policy?",
    "How many earned leaves are regular employees entitled to in a calendar year?",
    "How many leaves are given to an employee on their birthday?",
    "What happens if an employee remains absent for more than five days without informing the authorities?",
    "What are the company's working days and office timings?",
    "What happens if an employee repeatedly arrives late?",
   
]
ground_truth = [
    "The HR policy aims to provide continuity and consistency of service, better internal and external communication, enhanced orientation and focus, and mentoring reference.",
    "Regular employees are entitled to 18 earned leaves in a calendar year.",
    "Employees receive 0.5 leave on their birthday.",
    "An employee absent for more than five days without informing the concerned authorities may have their employment contract terminated.",
    "Employees work from Monday to Saturday. Office timings are 9:30 AM to 6:00 PM.",
    "Repeated late arrivals are treated as indiscipline and may lead to strict disciplinary action.",
]

def get_data():
   
    answer=[]
    context=[]
  

    for qs in questions:
        answer.append(chain.invoke(qs))
        rs=ret.invoke(qs)
        context.append([doc.page_content for doc in rs])

    dt=Dataset.from_dict({'question':questions, 'answer':answer, 'retrieved_contexts':context, 'ground_truth':ground_truth})
    return dt

data=get_data()

score=evaluate(dataset=data, metrics=[faithfulness, context_recall,context_precision,answer_relevancy], llm=llm, embeddings=emb)

df=score.to_pandas()

df.to_csv(r'D:\AI\VS CODE\Langchain\RAGAs\CompanyPolicy_RAG_Assistant_with_evaluation\bge-small.csv', index=False)
    


