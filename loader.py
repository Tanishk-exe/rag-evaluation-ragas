from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,PDFPlumberLoader


def load_file(name,path):
    if name=="PyPDF":
        loader=PyPDFLoader(path)
    elif name=="PyMu":
        loader=PyMuPDFLoader(path)
    elif name=="PDFPlumber":
        loader=PDFPlumberLoader(path)
    else:
        print("Input Inavlid!!!")
    docs=loader.load()
    return docs