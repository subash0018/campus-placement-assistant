import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from rag import build_graph

load_dotenv()
app = Flask(__name__)
graph = build_graph()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    tool_result = None
    question = ""
    form_values = {"cgpa": "", "active_backlogs": "", "department": ""}
    error = None

    if request.method == "POST":
        action = request.form.get("action", "ask_question")

        try:
            if action == "ask_question":
                question = request.form.get("question", "").strip()
                if not question:
                    raise ValueError("Please enter a question.")
                result = graph.invoke({
                    "action": "ask_question",
                    "question": question,
                    "context": "",
                    "sources": [],
                    "tool_inputs": {},
                    "tool_result": None,
                    "answer": ""
                })
                answer = result.get("answer")
                sources = result.get("sources", [])

            elif action == "match_companies":
                form_values = {
                    "cgpa": request.form.get("cgpa", "").strip(),
                    "active_backlogs": request.form.get("active_backlogs", "").strip(),
                    "department": request.form.get("department", "").strip()
                }
                try:
                    cgpa = float(form_values["cgpa"])
                    backlogs = int(form_values["active_backlogs"])
                except (ValueError, TypeError):
                    raise ValueError("CGPA must be a number and active backlogs must be a whole number.")

                result = graph.invoke({
                    "action": "match_companies",
                    "question": "",
                    "context": "",
                    "sources": [],
                    "tool_inputs": {
                        "cgpa": cgpa,
                        "active_backlogs": backlogs,
                        "department": form_values["department"]
                    },
                    "tool_result": None,
                    "answer": ""
                })
                tool_result = result.get("tool_result")
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Request failed")
            error = "Something went wrong. Check the terminal for details and verify your .env configuration."

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        tool_result=tool_result,
        question=question,
        form_values=form_values,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
