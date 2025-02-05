import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient
from opentelemetry import trace

from ragnar.config import settings
from ragnar.backend.tools import tracer
from ragnar.backend.tools import tavily_search_async
from ragnar.backend.utils import deduplicate_and_format_sources
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration


WRITING_INSTRUCTIONS = """You are an expert writer working on writing a summary about the following topic:

{topic}

Use this source material to help write the summary:

{context}

1. Highlight the most relevant information from each source
2. Provide a concise overview of the key points related to the report topic
3. Emphasize significant findings or insights
4. Ensure a coherent flow of information
"""

EXTENDING_INSTRUCTIONS = """You are an expert writer working on extending a research summary with new search results:

Topic of the summary:

{topic}

Existing summary:

{summary}

New search results:

{search_results}

1. Seamlessly integrate new information without repeating what's already covered
2. Maintain consistency with the existing content's style and depth
3. Only add new, non-redundant information
4. Ensure smooth transitions between existing and new content
"""


GUIDELINES = """

Guidelines for writing:

1. Length and Style:
- Maximum {word_count} words
- No marketing language
- Write in simple, clear language
- Start with your most important insight in **bold**

2. Structure:
- Use ## for summary title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the summary content

4. Quality Checks:
- Maximum {word_count} words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- Starts with bold insight
- No preamble prior to creating the summary content
- Sources cited at end
"""

class SummaryWriter:
    def __init__(self, model_name: str, context_window_length: int):
        self.model_name = model_name
        self.writer_llm = ChatOllama(model=model_name,
                                     temperature=0,
                                     base_url=settings.OLLAMA_URL,
                                     num_ctx=context_window_length)

    @tracer.start_as_current_span('summary_writer')
    def run(self, state: SummaryState) -> SummaryState:

        if state.summary_exists:
            # Extending existing summary
            instructions = (
                EXTENDING_INSTRUCTIONS.format(topic=state.topic, summary=state.content, search_results=state.source_str) +
                GUIDELINES.format(word_count=1000)
            )
        else:
            # Writing a new summary
            instructions = (
                    WRITING_INSTRUCTIONS.format(topic=state.topic, context=state.source_str) +
                    GUIDELINES.format(word_count=1000)
            )

        summary = self.writer_llm.invoke(
            [
                SystemMessage(content=instructions),
                # HumanMessage(content=human_message_content)
            ]
        )

        state.steps.append(Node.SUMMARY_WRITER.value)
        state.content = summary.content
        state.summary_exists = True

        span = trace.get_current_span()
        span.set_status(trace.StatusCode.OK)
        span.set_attributes(
            attributes={
                'topic': state.topic,
                'model_name': self.model_name,
                'iteration': state.iteration,
                'summary': state.content,
            }
        )

        return state
