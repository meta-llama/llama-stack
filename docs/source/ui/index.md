# Playground UI

```{note}
Playground UI is currently experimental and subject to change. We welcome feedback and contributions to help improve it.
```

Llama Stack Playground UI is an simple interface aims to:
- Showcase **capabilities** and **concepts** of Llama Stack in an interactive environment
- Demo **end-to-end** application code to help users get started to build their own applications
- Provide an **UI** to help users inspect and analyze Llama Stack API providers and resources

## Key Features

#### Playground
Interactive pages for users to play with and explore Llama Stack API capabilities.

##### Chatbot
- **Chat**: Chat with Llama models
- **RAG**: Uploading documents to memory_banks and chat with RAG agent
```{eval-rst}
.. video:: ../../resources/video/chat.mov
    :nocontrols:
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```

##### Evaluations
- **Evaluations (Scoring)**: Run evaluations on your AI application datasets

```{eval-rst}
.. video:: ../../resources/video/scoring.mov
    :nocontrols:
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```


- **Evaluations (Generation + Scoring)**: Use pre-registered evaluation tasks to evaluate an model or agent candidate.
```{eval-rst}
.. video:: ../../resources/video/gen_scoring.mov
    :nocontrols:
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```

##### Inspect
Inspect Llama Stack API providers and resources (models, datasets, memory_banks, eval_tasks, etc).
```{eval-rst}
.. video:: ../../resources/video/inspect.mov
    :nocontrols:
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```

## Starting the Playground UI

To start the Playground UI, run the following commands:

1. Start up the Llama Stack API server

```bash
llama stack build --template together --image-type conda
llama stack run together
```

2. (Optional) Register datasets and eval tasks as resources. If you want to run pre-configured evaluation flows (e.g. Evaluations (Generation + Scoring) Page).
```bash
$ llama-stack-client datasets register \
--dataset-id "mmlu" \
--provider-id "huggingface" \
--url "https://huggingface.co/datasets/llamastack/evals" \
--metadata '{"path": "llamastack/evals", "name": "evals__mmlu__details", "split": "train"}' \
--schema '{"input_query": {"type": "string"}, "expected_answer": {"type": "string", "chat_completion_input": {"type": "string"}}}'
```

```bash
$ llama-stack-client eval_tasks register \
--eval-task-id meta-reference-mmlu \
--provider-id meta-reference \
--dataset-id mmlu \
--scoring-functions basic::regex_parser_multiple_choice_answer
```

3. Start Streamlit UI
```bash
cd llama_stack/distribution/ui
pip install -r requirements.txt
streamlit run app.py
```
