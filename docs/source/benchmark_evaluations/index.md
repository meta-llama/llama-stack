# Benchmark Evaluations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10CHyykee9j2OigaIcRv47BKG9mrNm0tJ?usp=sharing)

Llama Stack provides the building blocks needed to run benchmark and application evaluations. This guide will walk you through how to use these components to run open benchmark evaluations. Visit our [Evaluation Concepts](../concepts/evaluation_concepts.md) guide for more details on how evaluations work in Llama Stack, and our [Evaluation Reference](../references/evals_reference/index.md) guide for a comprehensive reference on the APIs. Check out our [Colab notebook](https://colab.research.google.com/drive/10CHyykee9j2OigaIcRv47BKG9mrNm0tJ?usp=sharing) on working examples on how you can use Llama Stack for running benchmark evaluations.

### 1. Open Benchmark Model Evaluation

This first example walks you through how to evaluate a model candidate served by Llama Stack on open benchmarks. We will use the following benchmark:
- [MMMU](https://arxiv.org/abs/2311.16502) (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI): Benchmark designed to evaluate multimodal models.
- [SimpleQA](https://openai.com/index/introducing-simpleqa/): Benchmark designed to access models to answer short, fact-seeking questions.

#### 1.1 Running MMMU
- We will use a pre-processed MMMU dataset from [llamastack/mmmu](https://huggingface.co/datasets/llamastack/mmmu). The preprocessing code is shown in in this [Github Gist](https://gist.github.com/yanxi0830/118e9c560227d27132a7fd10e2c92840). The dataset is obtained by transforming the original [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU) dataset into correct format by `inference/chat-completion` API.

```python
import datasets
ds = datasets.load_dataset(path="llamastack/mmmu", name="Agriculture", split="dev")
ds = ds.select_columns(["chat_completion_input", "input_query", "expected_answer"])
eval_rows = ds.to_pandas().to_dict(orient="records")
```

- Next, we will run evaluation on an model candidate, we will need to:
  - Define a system prompt
  - Define an EvalCandidate
  - Run evaluate on the dataset

```python
SYSTEM_PROMPT_TEMPLATE = """
You are an expert in Agriculture whose job is to answer questions from the user using images.
First, reason about the correct answer.
Then write the answer in the following format where X is exactly one of A,B,C,D:
Answer: X
Make sure X is one of A,B,C,D.
If you are uncertain of the correct answer, guess the most likely one.
"""

system_message = {
    "role": "system",
    "content": SYSTEM_PROMPT_TEMPLATE,
}

client.eval_tasks.register(
    eval_task_id="meta-reference::mmmu",
    dataset_id=f"mmmu-{subset}-{split}",
    scoring_functions=["basic::regex_parser_multiple_choice_answer"]
)

response = client.eval.evaluate_rows(
    task_id="meta-reference::mmmu",
    input_rows=eval_rows,
    scoring_functions=["basic::regex_parser_multiple_choice_answer"],
    task_config={
        "type": "benchmark",
        "eval_candidate": {
            "type": "model",
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "sampling_params": {
                "strategy": {
                    "type": "greedy",
                },
                "max_tokens": 4096,
                "repeat_penalty": 1.0,
            },
            "system_message": system_message
        }
    }
)
```

#### 1.2. Running SimpleQA
- We will use a pre-processed SimpleQA dataset from [llamastack/evals](https://huggingface.co/datasets/llamastack/evals/viewer/evals__simpleqa) which is obtained by transforming the input query into correct format accepted by `inference/chat-completion` API.
- Since we will be using this same dataset in our next example for Agentic evaluation, we will register it using the `/datasets` API, and interact with it through `/datasetio` API.

```python
simpleqa_dataset_id = "huggingface::simpleqa"

_ = client.datasets.register(
    dataset_id=simpleqa_dataset_id,
    provider_id="huggingface",
    url={"uri": "https://huggingface.co/datasets/llamastack/evals"},
    metadata={
        "path": "llamastack/evals",
        "name": "evals__simpleqa",
        "split": "train",
    },
    dataset_schema={
        "input_query": {"type": "string"},
        "expected_answer": {"type": "string"},
        "chat_completion_input": {"type": "chat_completion_input"},
    }
)

eval_rows = client.datasetio.get_rows_paginated(
    dataset_id=simpleqa_dataset_id,
    rows_in_page=5,
)
```

```python
client.eval_tasks.register(
    eval_task_id="meta-reference::simpleqa",
    dataset_id=simpleqa_dataset_id,
    scoring_functions=["llm-as-judge::405b-simpleqa"]
)

response = client.eval.evaluate_rows(
    task_id="meta-reference::simpleqa",
    input_rows=eval_rows.rows,
    scoring_functions=["llm-as-judge::405b-simpleqa"],
    task_config={
        "type": "benchmark",
        "eval_candidate": {
            "type": "model",
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "sampling_params": {
                "strategy": {
                    "type": "greedy",
                },
                "max_tokens": 4096,
                "repeat_penalty": 1.0,
            },
        }
    }
)
```


### 2. Agentic Evaluation
- In this example, we will demonstrate how to evaluate a agent candidate served by Llama Stack via `/agent` API.
- We will continue to use the SimpleQA dataset we used in previous example.
- Instead of running evaluation on model, we will run the evaluation on a Search Agent with access to search tool. We will define our agent evaluation candidate through `AgentConfig`.

```python
agent_config = {
    "model": "meta-llama/Llama-3.1-405B-Instruct",
    "instructions": "You are a helpful assistant",
    "sampling_params": {
        "strategy": {
            "type": "greedy",
        },
    },
    "tools": [
        {
            "type": "brave_search",
            "engine": "tavily",
            "api_key": userdata.get("TAVILY_SEARCH_API_KEY")
        }
    ],
    "tool_choice": "auto",
    "tool_prompt_format": "json",
    "input_shields": [],
    "output_shields": [],
    "enable_session_persistence": False
}

response = client.eval.evaluate_rows(
    task_id="meta-reference::simpleqa",
    input_rows=eval_rows.rows,
    scoring_functions=["llm-as-judge::405b-simpleqa"],
    task_config={
        "type": "benchmark",
        "eval_candidate": {
            "type": "agent",
            "config": agent_config,
        }
    }
)
```
