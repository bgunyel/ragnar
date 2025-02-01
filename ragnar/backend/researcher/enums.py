from enum import Enum


class Node(Enum):
    # In alphabetical order
    PLANNER = 'planner'
    QUERY_WRITER = 'query_writer'
    RESET = 'reset'
    SUMMARY_WRITER = 'summary_writer'
    SUMMARY_REVIEWER = 'summary_reviewer'
    WEB_SEARCH = 'web_search'

