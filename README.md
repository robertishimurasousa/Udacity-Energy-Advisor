# EcoHome Energy Advisor

A Udacity project to build a LangGraph agent that helps households decide when to use electricity, how to make the most of rooftop solar, and how much they could save. The agent will combine energy history, weather forecasts, time-of-use rates, and energy-saving guidance to recommend actions for EV charging, heating and cooling, and household appliances.

This is the Energy Advisor project for course 2 of 3 in the Udacity Nanodegree. [Building AI Agents with LangGraph](https://www.udacity.com/course/building-ai-agents-with-langgraph--cd14640) · [Agentic AI Engineer with LangChain and LangGraph](https://www.udacity.com/course/agentic-ai-engineer-with-langchain-and-langgraph--nd901)

**Status:** This repository contains the Udacity starter files. Implementation and notebook validation are pending.

## Pick up on another Mac

```bash
git clone https://github.com/robertishimurasousa/Udacity-Energy-Advisor.git
cd Udacity-Energy-Advisor
```

The starter files currently live at the repository root. Before implementing the assignment, copy the starter into `ecohome_solution/` and use that directory for development and submission.

The notebooks record Python 3.11.10 in their metadata. The starter includes two dependency manifests. Confirm a working environment and document the Python and package versions used for the submission. The starter model and embedding setup read `VOCAREUM_API_KEY` from the environment. Keep that credential local. [requirements.txt](requirements.txt) · [requirements 2.txt](requirements%202.txt)

## Starter files

| File or directory | Purpose |
| --- | --- |
| [`01_db_setup.ipynb`](01_db_setup.ipynb) | Create and populate SQLite with sample consumption and solar generation data. |
| [`02_rag_setup.ipynb`](02_rag_setup.ipynb) | Load, split, and embed energy advice documents, then index them in Chroma. |
| [`03_run_and_evaluate.ipynb`](03_run_and_evaluate.ipynb) | Define the agent's instructions, run scenarios, and evaluate responses and tool calls. |
| [`agent.py`](agent.py) and [`tools.py`](tools.py) | Agent interface, graph, and tools for data retrieval and recommendations. |
| [`models/energy.py`](models/energy.py) | SQLAlchemy models and database access for energy usage and solar generation. |
| [`data/documents/`](data/documents/) | Two starter documents covering general energy savings and device best practices. |

## Assignment milestones

1. **Prepare the data.** Run the database notebook and verify both energy usage and solar generation tables. Add at least five documents covering HVAC, home automation, renewables, seasonal energy management, and battery storage. Run the RAG notebook and confirm the added documents can be retrieved.
2. **Implement the tools.** Complete `get_weather_forecast`, `get_electricity_prices`, and `search_energy_tips`. Simulated forecasts and rates are allowed. The tips tool should create or reopen the Chroma index and return structured search results.
3. **Build the agent.** Define the model and LangGraph workflow in `agent.py`. Preserve `Agent(instructions, model)` and `invoke(question, context)`. Write `ECOHOME_SYSTEM_PROMPT` in the evaluation notebook and use it to guide tool selection, error handling, recommendations, source citations, and savings estimates.
4. **Evaluate the end-to-end flow.** Run at least ten scenarios and record responses and tool calls in `test_results`. Evaluate accuracy, relevance, completeness, and usefulness of responses, as well as appropriateness and completeness of tool usage. Display a report with scores, feedback, and proposed improvements.

## First study case

> How much can I save by running my dishwasher during off-peak hours?

Compare the cost of the same cycle at peak and off-peak rates. With assumed rates and an assumed energy use per cycle, the estimate is **energy per cycle × (peak rate − off-peak rate)**. This gives you an expected result to check before evaluating the agent's recommendation.
