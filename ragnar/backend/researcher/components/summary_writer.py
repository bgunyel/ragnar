import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from ragnar.config import settings
from ragnar.backend.tools import tavily_search_async
from ragnar.backend.utils import deduplicate_and_format_sources
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration


SUMMARY_WRITER_INSTRUCTIONS = """You are an expert writer working on writing a summary about the following topic:

{topic}

Use this source material to help write the summary:

{context}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Approximately {word_limit} words
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**

3. Structure:
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

4. Writing Approach:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the summary content
- Focus on your single most important point

5. Quality Checks:
- Approximately {word_limit} words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the summary content
- Sources cited at end
"""

class SummaryWriter:
    def __init__(self, model_name: str):
        self.writer_llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL)

    def run(self, state: SummaryState) -> SummaryState:

        if state.summary_exists: # Extending existing summary
            human_message_content = (
                f"Extend the existing summary: {state.content}\n\n"
                f"Include new search results: {state.source_str} "
                f"That addresses the following topic: {state.topic}"
            )
            instructions = SUMMARY_WRITER_INSTRUCTIONS
        else:
            # Writing a new summary

            instructions = SUMMARY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                              context=state.source_str,
                                                              word_limit=1000)

            # human_message_content = "Generate a summary based on the provided sources."


        summary = self.writer_llm.invoke(
            [
                SystemMessage(content=instructions),
                # HumanMessage(content=human_message_content)
            ]
        )

        state.steps.append(Node.SUMMARY_WRITER.value)
        state.content = summary.content
        state.summary_exists = True

        return state
