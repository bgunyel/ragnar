import asyncio
from tavily import AsyncTavilyClient

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

from ragnar.config import settings
from ragnar.backend.base import TavilySearchCategory



###
# Configure OpenTelemetry
###

resource = Resource(
    attributes={
        ResourceAttributes.SERVICE_NAME: settings.APPLICATION_NAME,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
    }
)
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")) # OTLP endpoint for Jaeger
provider.add_span_processor(processor)
trace.set_tracer_provider(provider) # Sets the global default tracer provider
tracer = trace.get_tracer(f'{settings.APPLICATION_NAME}.tracer') # Creates a tracer from the global tracer provider


async def tavily_search_async(client: AsyncTavilyClient,
                              search_queries: list[str],
                              search_category: TavilySearchCategory,
                              number_of_days_back: int,
                              max_results: int = 5):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        client: Async Tavily Client
        search_queries (Queries): List of search queries to process
        search_category (str): Type of search to perform ('news' or 'general')
        number_of_days_back (int): Number of days to look back for news articles (only used when tavily_topic='news')
        max_results (int): The maximum number of search results to return. Default is 5.

    Returns:
        List[dict]: List of search results from Tavily API, one per query

    Note:
        For news searches, each result will include articles from the last `number_of_days_back` days.
        For general searches, the time range is unrestricted.
    """

    kwargs = {
        'max_results': max_results,
        'include_raw_content': True,
        'topic': search_category,
    }
    if search_category == 'news':
        kwargs['days'] = number_of_days_back

    # Execute all searches concurrently
    search_tasks = [client.search(query=query, **kwargs) for query in search_queries]
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs
