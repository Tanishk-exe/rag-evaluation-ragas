

def get_ret(vs,k=4):
    return vs.as_retriever(search_kwargs={"k":k})