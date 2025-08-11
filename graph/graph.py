from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.route_grader import question_router, RouterQuery
from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

# Tools (Edges)
def route_question(state: GraphState):
    question = state["question"]

    get_route_question_result = question_router.invoke({"question": question})
    if get_route_question_result.datasource == "vectorstore":
        return RETRIEVE
    elif get_route_question_result.datasource == WEBSEARCH:
        return WEBSEARCH

def is_relevant_docs(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "NO"
    else:
        print("---DECISION: GENERATE---")
        return "YES"

def grade_generation_v_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # hallucination_grade = score.binary_score
    # if hallucination_grade is not None:
    #   ....

    # Check hallucination
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# Graph Structure
builder = StateGraph(GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEBSEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.set_conditional_entry_point(
    route_question, # Condition
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    }
)

builder.add_edge(WEBSEARCH, GENERATE)

builder.add_edge(RETRIEVE, GRADE_DOCUMENTS) # Direct edge

builder.add_conditional_edges(
    GRADE_DOCUMENTS, # Node
    is_relevant_docs, # Take node result and grade it.
    {
        "NO": WEBSEARCH,
        "YES": GENERATE,
    }
)

builder.add_conditional_edges(
    GENERATE,
    grade_generation_v_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

builder.add_edge(GENERATE, END)

app = builder.compile()