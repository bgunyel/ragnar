from enum import Enum

class Node(Enum):
    # In alphabetical order
    ANSWER_GENERATOR = 'answer_generator'
    ANSWER_GRADER = 'answer_grader'
    DOCUMENT_GRADER = 'document_grader'
    HALLUCINATION_GRADER = 'hallucination_grader'
    INCREMENT_ITERATION = 'increment_iteration'
    INTERNAL_ANSWER_GENERATOR = 'internal_answer_generator'
    RESET = 'reset'
    RETRIEVE = 'retrieve'
    REWRITE_QUESTION = 'rewrite_question'
    ROUTER = 'router'

class Grades(Enum):
    GOOD = 'Good'
    BAD = 'Bad'

"""

class StateField(Enum):
    # Alphabetical Order
    ANSWER_GROUNDED = "answer_grounded"
    ANSWER_USEFUL = "answer_useful"
    DATA_SOURCE = 'datasource'
    DOCUMENTS = 'documents'
    DOCUMENTS_GRADE = 'documents_grade'
    GENERATION = 'generation'
    GENERATION_ITERATION = "generation_iteration"
    GOOD_DOCUMENTS = 'good_documents'
    ORIGINAL_QUESTION = 'original_question'
    QUESTION = "question"
    RETRIEVAL_ITERATION = "retrieval_iteration"
    STEPS = "steps"
    
"""
