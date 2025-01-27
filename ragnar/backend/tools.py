import asyncio
from typing import Literal
from tavily import AsyncTavilyClient

from ragnar.backend.base import SearchQuery


async def tavily_search_async(client: AsyncTavilyClient,
                              search_queries: list[SearchQuery],
                              search_category: Literal['news', 'general'],
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
    search_tasks = [client.search(query=query.search_query, **kwargs) for query in search_queries]
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs
