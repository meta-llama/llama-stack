## Agentic API 101

This document talks about the Agentic APIs in Llama Stack.

Starting Llama 3.1 you can build agentic applications capable of:

- breaking a task down and performing multi-step reasoning.
- using tools to perform some actions
  - built-in: the model has built-in knowledge of tools like search or code interpreter
  - zero-shot: the model can learn to call tools using previously unseen, in-context tool definitions
- providing system level safety protections using models like Llama Guard.

An agentic app requires a few components:
- ability to run inference on the underlying Llama series of models
- ability to run safety checks using the Llama Guard series of models
- ability to execute tools, including a code execution environment, and loop using the model's multi-step reasoning process

All of these components are now offered by a single Llama Stack Distribution. Llama Stack defines and standardizes these components and many others that are needed to make building Generative AI applications smoother. Various implementations of these APIs are then assembled together via a **Llama Stack Distribution**.

### Run Agent example

To run an agent app, check out examples demo scripts with client SDKs to talk with the Llama Stack server in our llama-stack-apps repo. With the server running, to run a simple agent app:

```bash
git clone git@github.com:meta-llama/llama-stack-apps.git
cd llama-stack-apps
pip install -r requirements.txt

python -m examples.agents.client <host> <port>
```

You will see outputs like this:

```bash
User> I am planning a trip to Switzerland, what are the top 3 places to visit?
inference> Switzerland is a beautiful country with a rich history, stunning landscapes, and vibrant culture. Here are three must-visit places to add to your itinerary:
...

User> What is so special about #1?
inference> Jungfraujoch, also known as the "Top of Europe," is a unique and special place for several reasons:
...

User> What other countries should I consider to club?
inference> Considering your interest in Switzerland, here are some neighboring countries that you may want to consider visiting:
```






## Readme for llama-stack-app:


### Agentic System Concept

![Figure 2: Agentic System](../docs/resources/agentic-system.png)

In addition to the model lifecycle, we considered the different components involved in an agentic system. Specifically around tool calling and shields. Since the model may decide to call tools, a single model inference call is not enough. What’s needed is an agentic loop consisting of tool calls and inference. The model provides separate tokens representing end-of-message and end-of-turn. A message represents a possible stopping point for execution where the model can inform the execution environment that a tool call needs to be made. The execution environment, upon execution, adds back the result to the context window and makes another inference call. This process can get repeated until an end-of-turn token is generated.
Note that as of today, in the OSS world, such a “loop” is often coded explicitly via elaborate prompt engineering using a ReAct pattern (typically) or preconstructed execution graph. Llama 3.1 (and future Llamas) attempts to absorb this multi-step reasoning loop inside the main model itself.

**Let's consider an example:**
1. The user asks the system "Who played the NBA finals last year?"
2. The model "understands" that this question needs to be answered using web search. It answers this abstractly with a message of the form "Please call the search tool for me with the query: 'List finalist teams for NBA in the last year' ". Note that the model by itself does not call the tool (of course!)
3. The executor consults the set of tool implementations which have been configured by the developer to find an implementation for the "search tool". If it does not find it, it returns an error to the model. Otherwise, it executes this tool and returns the result of this tool back to the model.
4. The model reasons once again (using all the messages above) and decides to send a final response "In 2023, Denver Nuggets played against the Miami Heat in the NBA finals." to the executor
6. The executor returns the response directly to the user (since there is no tool call to be executed.)

The sequence diagram that details the steps is [here](https://github.com/meta-llama/llama-agentic-system/blob/main/docs/sequence-diagram.md).

* /memory_banks - to support creating multiple repositories of data that can be available for agentic systems
* /agentic_system - to support creating and running agentic systems. The sub-APIs support the creation and management of the steps, turns, and sessions within agentic applications.
  * /step - there can be inference, memory retrieval, tool call, or shield call steps
  * /turn - each turn begins with a user message and results in a loop consisting of multiple steps, followed by a response back to the user
  * /session - each session consists of multiple turns that the model is reasoning over
  * /memory_bank - a memory bank allows for the agentic system to perform retrieval augmented generation




### How to build your own agent

Agents Protocol is defined in [agents.py](../llama_stack/apis/agents/agents.py). Your agent class must have the following functions:

**create_agent(agent_config)**:

**create_agent_turn(agent_id,session_id,messages)**:

**get_agents_turn(agent_id, session_id, turn_id)**:

**get_agents_step(agent_id, session_id, turn_id, step_id)**:

**create_agent_session(agent_id, session_id)**:

**get_agents_session(agent_id, session_id, turn_id)**:

**delete_agents_session(agent_id, session_id)**:

**delete_agents(agent_id, session_id)**:
