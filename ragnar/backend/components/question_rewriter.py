from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragnar.config import settings


def get_question_rewriter(model_name: str):
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL)
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

"""
TEST:
question_rewriter.invoke({"question": question})
"""
