from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)

def get_rag_generator(model_name: str):
    llm = ChatOllama(model=model_name, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

"""
# Test
generation = rag_chain.invoke({"documents": docs, "question": question})
print(generation)
"""