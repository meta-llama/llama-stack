# Evaluations

The Llama Stack provides a set of APIs in Llama Stack for supporting running evaluations of LLM applications.
- `/datasetio` + `/datasets` API
- `/scoring` + `/scoring_functions` API
- `/eval` + `/benchmarks` API



This guides walks you through the process of evaluating an LLM application built using Llama Stack. Checkout the [Evaluation Reference](../references/evals_reference/index.md) guide goes over the sets of APIs and developer experience flow of using Llama Stack to run evaluations for benchmark and application use cases. Checkout our Colab notebook on working examples with evaluations [here](https://colab.research.google.com/drive/10CHyykee9j2OigaIcRv47BKG9mrNm0tJ?usp=sharing).


## Application Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb)

Llama Stack offers a library of scoring functions and the `/scoring` API, allowing you to run evaluations on your pre-annotated AI application datasets.

In this example, we will show you how to:
1. Build an Agent with Llama Stack
2. Query the agent's sessions, turns, and steps
3. Evaluate the results.

##### Building a Search Agent
```python
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

client = LlamaStackClient(base_url=f"http://{HOST}:{PORT}")

agent = Agent(
    client,
    model="meta-llama/Llama-3.3-70B-Instruct",
    instructions="You are a helpful assistant. Use search tool to answer the questions. ",
    tools=["builtin::websearch"],
)
user_prompts = [
    "Which teams played in the NBA Western Conference Finals of 2024. Search the web for the answer.",
    "In which episode and season of South Park does Bill Cosby (BSM-471) first appear? Give me the number and title. Search the web for the answer.",
    "What is the British-American kickboxer Andrew Tate's kickboxing name? Search the web for the answer.",
]

session_id = agent.create_session("test-session")

for prompt in user_prompts:
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        session_id=session_id,
    )

    for log in AgentEventLogger().log(response):
        log.print()
```


##### Query Agent Execution Steps

Now, let's look deeper into the agent's execution steps and see if how well our agent performs.
```python
# query the agents session
from rich.pretty import pprint

session_response = client.agents.session.retrieve(
    session_id=session_id,
    agent_id=agent.agent_id,
)

pprint(session_response)
```

As a sanity check, we will first check if all user prompts is followed by a tool call to `brave_search`.
```python
num_tool_call = 0
for turn in session_response.turns:
    for step in turn.steps:
        if (
            step.step_type == "tool_execution"
            and step.tool_calls[0].tool_name == "brave_search"
        ):
            num_tool_call += 1

print(
    f"{num_tool_call}/{len(session_response.turns)} user prompts are followed by a tool call to `brave_search`"
)
```

##### Evaluate Agent Responses
Now, we want to evaluate the agent's responses to the user prompts.

1. First, we will process the agent's execution history into a list of rows that can be used for evaluation.
2. Next, we will label the rows with the expected answer.
3. Finally, we will use the `/scoring` API to score the agent's responses.

```python
eval_rows = []

expected_answers = [
    "Dallas Mavericks and the Minnesota Timberwolves",
    "Season 4, Episode 12",
    "King Cobra",
]

for i, turn in enumerate(session_response.turns):
    eval_rows.append(
        {
            "input_query": turn.input_messages[0].content,
            "generated_answer": turn.output_message.content,
            "expected_answer": expected_answers[i],
        }
    )

pprint(eval_rows)

scoring_params = {
    "basic::subset_of": None,
}
scoring_response = client.scoring.score(
    input_rows=eval_rows, scoring_functions=scoring_params
)
pprint(scoring_response)
```
