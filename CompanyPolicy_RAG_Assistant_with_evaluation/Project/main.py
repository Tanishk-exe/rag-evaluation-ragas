from model import get_model
from loader import load_file
from splitter import split
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from embedding import get_emb
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from retreiver import get_ret

model=get_model("Gemini")
emb=get_emb()
path="D:\AI\VS CODE\Langchain\RAGAs\CompanyPolicy_RAG_Assistant_with_evaluation\HR-Policy.pdf"
docs=load_file(name="PyMu",path=path)
sp_docs=split(docs,750,100)

vs=FAISS.from_documents(documents=sp_docs,embedding=emb)
ret=get_ret(vs,k=6)

def ret_invoke(query):
    return ret.invoke(query)

#Static Code(NO CHANGE!!!)
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
parser=StrOutputParser()

def chain_invoke(query):
    paral=RunnableParallel(question= RunnablePassthrough(),context=ret )
    seq=temp | model | parser
    chain=paral | seq
    return chain.invoke(query)




