from typing import TypedDict
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from ragnar.config import settings

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

def get_answer_generator(model_name: str):
    llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain


class AnswerGenerator:
    def __init__(self, model_name: str):
        self.rag_generator = get_answer_generator(model_name=model_name)

    def run(self, state: TypedDict) -> TypedDict:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        context = "\n\n".join(doc.page_content for doc in state['good_documents'])
        state['generation'] = self.rag_generator.invoke({'documents': context, 'question': state['original_question']})
        state['steps'].append("generate_answer")
        return state