from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from ragnar.config import settings
from ragnar.backend.enums import Node
from ragnar.backend.state import GraphState

rag_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Keep the answer concise:
    
    Question: {question} 
    
    Documents: {documents}
     
    Answer: 
    """,
    input_variables=["question", "documents"],
)

internal_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Answer the following question. 

    If you don't know the answer, just say that you don't know. 

    Keep the answer concise:

    Question: {question}

    Answer: 
    """,
    input_variables=["question"],
)

def get_answer_generator(model_name: str):
    llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL)
    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain

def get_internal_answer_generator(model_name: str):
    llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL)
    chain = internal_prompt | llm | StrOutputParser()
    return chain


class AnswerGenerator:
    def __init__(self, model_name: str, is_rag:bool = True):
        self.is_rag = is_rag
        self.generator = None

        if self.is_rag:
            self.generator = get_answer_generator(model_name=model_name)
        else:
            self.generator = get_internal_answer_generator(model_name=model_name)


    def run(self, state: GraphState) -> GraphState:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        state.generation_iteration += 1

        if self.is_rag:
            context = "\n\n".join(doc.page_content for doc in state.good_documents)
            state.generation = self.generator.invoke({'documents': context, 'question': state.question})
            state.steps.append(Node.ANSWER_GENERATOR.value)
        else:
            state.generation = self.generator.invoke({'question': state.question})
            state.steps.append(Node.INTERNAL_ANSWER_GENERATOR.value)

        return state
