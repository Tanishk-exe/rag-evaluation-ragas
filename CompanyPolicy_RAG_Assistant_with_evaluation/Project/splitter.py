from langchain_text_splitters import RecursiveCharacterTextSplitter


def split(doc, chunk_size,chunk_overlap):
    sp=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n","\n",". "," ",""])
    splitted=sp.split_documents(doc)
    return splitted