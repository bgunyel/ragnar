from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from ragnar.config import settings
from ragnar.backend.state import GraphState
from ragnar.backend.enums import Node


query_writer_instructions="""Your goal is to generate targeted web search query.

The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string",
    "aspect": "string",
    "rationale": "string"
}}
"""


prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version that is optimized for web search.
    Look at the input and try to reason about the underlying semantic intent / meaning. Do not extend the scope of the original question.
    
    "Here is the initial question: \n\n {question}
    \n\n
    Formulate an improved question.
    
    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct.
    
    Return your improved question and rationale as a JSON object:
    {{
    "improved_question": "string",
    "rationale": "string"
    }}
    """,
    input_variables=['question']
)



def get_question_rewriter(model_name: str):
    llm = ChatOllama(model=model_name, format="json", temperature=0, base_url=settings.OLLAMA_URL)
    question_rewriter = prompt | llm | JsonOutputParser()
    return question_rewriter


class QuestionRewriter:
    def __init__(self, model_name: str):
        self.question_rewriter = get_question_rewriter(model_name=model_name)

    def run(self, state: GraphState) -> GraphState:
        # Re-write question
        new_question = self.question_rewriter.invoke({"question": state.query})
        state.query = new_question['improved_question']
        state.steps.append(Node.REWRITE_QUESTION.value)
        return state
