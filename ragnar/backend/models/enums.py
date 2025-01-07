from enum import Enum

class Nodes(Enum):
    # In alphabetical order
    GENERATE = 'generate'
    GRADE_DOCS = 'grade_documents'
    INCREMENT_ITERATION = 'increment_iteration'
    RETRIEVE = 'retrieve'
    REWRITE_QUESTION = 'rewrite_question'

class Grades(Enum):
    GOOD = 'Good'
    BAD = 'Bad'

class States(Enum):
    QUESTION = "question"
    DOCUMENTS = 'documents'
    GENERATION = 'generation'
    DOCUMENTS_GRADE = 'documents_grade'
    STEPS = "steps"
    ITERATION = "iteration"