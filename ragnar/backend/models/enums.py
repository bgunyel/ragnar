from enum import Enum

class Nodes(Enum):
    # In alphabetical order
    GENERATE = 'generate'
    GRADE_DOCS = 'grade_documents'
    RETRIEVE = 'retrieve'
    REWRITE_QUESTION = 'rewrite_question'