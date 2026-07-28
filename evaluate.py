from ragas import evaluate
from ragas.metrics import faithfulness,context_precision,context_recall,answer_relevancy
from datasets import Dataset
from main import chain_invoke, ret_invoke


qs = [
    "What is the objective of the HR policy?",
    "How many earned leaves are regular employees entitled to in a calendar year?",
    "How many leaves are given to an employee on their birthday?",
    "What happens if an employee remains absent for more than five days without informing the authorities?",
    "What are the company's working days and office timings?",
    "What happens if an employee repeatedly arrives late?",
   
]
gt = [
    "The HR policy aims to provide continuity and consistency of service, better internal and external communication, enhanced orientation and focus, and mentoring reference.",
    "Regular employees are entitled to 18 earned leaves in a calendar year.",
    "Employees receive 0.5 leave on their birthday.",
    "An employee absent for more than five days without informing the concerned authorities may have their employment contract terminated.",
    "Employees work from Monday to Saturday. Office timings are 9:30 AM to 6:00 PM.",
    "Repeated late arrivals are treated as indiscipline and may lead to strict disciplinary action.",
]

ans=[]
ct=[]

for q in qs:
    ans.append(chain_invoke(q))
    rs=ret_invoke(q)
    r=[doc.page_content for doc in rs]
    ct.append(r)


def get_data():
    return Dataset.from_dict({"question":qs,"answer":ans,"retrieved_context":ct,"ground_truth":gt})

data=get_data()

score=evaluate(dataset=data, metrics=[faithfulness,context_recall,context_precision,answer_relevancy])

sc=score.to_pandas()

from pathlib import Path
name =input("Model/Embedding name: ")
output_path = Path("D:/AI/VS CODE/Langchain\RAGAs/CompanyPolicy_RAG_Assistant_with_evaluation/project") / f"{name}.csv"
sc.to_csv(output_path, index=False)
