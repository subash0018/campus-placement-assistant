# Learn more about building a Python provider: https://promptfoo.dev/docs/providers/python/

"""Promptfoo provider for the Campus Placement Assistant."""

from rag import build_graph

# Reuse the existing Campus Placement Assistant workflow.
graph = build_graph()


def call_api(prompt, options, context):
    """Send a test question through the existing RAG pipeline."""

    try:
        variables = context.get("vars", {})
        question = variables.get("question", "").strip()

        if not question:
            question = prompt.strip()

        if not question:
            return {"error": "No question was provided."}

        result = graph.invoke({
            "action": "ask_question",
            "question": question,
            "context": "",
            "sources": [],
            "tool_inputs": {},
            "tool_result": None,
            "answer": ""
        })

        return {
            "output": result.get(
                "answer",
                "No answer was generated."
            )
        }

    except Exception as exc:
        return {"error": str(exc)}