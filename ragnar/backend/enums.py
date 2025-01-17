from enum import Enum

class Node(Enum):
    # In alphabetical order
    ANSWER_GENERATOR = 'answer_generator'
    ANSWER_GRADER = 'answer_grader'
    DOCUMENT_GRADER = 'document_grader'
    HALLUCINATION_GRADER = 'hallucination_grader'
    INCREMENT_ITERATION = 'increment_iteration'
    INTERNAL_ANSWER_GENERATOR = 'internal_answer_generator'
    RETRIEVE = 'retrieve'
    REWRITE_QUESTION = 'rewrite_question'
    ROUTER = 'router'

class Grades(Enum):
    GOOD = 'Good'
    BAD = 'Bad'

class StateField(Enum):
    QUESTION = "question"
    DOCUMENTS = 'documents'
    GENERATION = 'generation'
    DOCUMENTS_GRADE = 'documents_grade'
    STEPS = "steps"
    ITERATION = "iteration"
    ANSWER_GROUNDED = "answer_grounded"
    ANSWER_USEFUL = "answer_useful"
    DATA_SOURCE = 'datasource'
    GOOD_DOCUMENTS = 'good_documents'

