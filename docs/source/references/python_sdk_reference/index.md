# Python SDK Reference

## Shared Types

```python
from llama_stack_client.types import (
    Attachment,
    BatchCompletion,
    CompletionMessage,
    SamplingParams,
    SystemMessage,
    ToolCall,
    ToolResponseMessage,
    UserMessage,
)
```

## Telemetry

Types:

```python
from llama_stack_client.types import TelemetryGetTraceResponse
```

Methods:

- <code title="get /telemetry/get_trace">client.telemetry.<a href="./src/llama_stack_client/resources/telemetry.py">get_trace</a>(\*\*<a href="src/llama_stack_client/types/telemetry_get_trace_params.py">params</a>) -> <a href="./src/llama_stack_client/types/telemetry_get_trace_response.py">TelemetryGetTraceResponse</a></code>
- <code title="post /telemetry/log_event">client.telemetry.<a href="./src/llama_stack_client/resources/telemetry.py">log</a>(\*\*<a href="src/llama_stack_client/types/telemetry_log_params.py">params</a>) -> None</code>

## Agents

Types:

```python
from llama_stack_client.types import (
    InferenceStep,
    MemoryRetrievalStep,
    RestAPIExecutionConfig,
    ShieldCallStep,
    ToolExecutionStep,
    ToolParamDefinition,
    AgentCreateResponse,
)
```

Methods:

- <code title="post /agents/create">client.agents.<a href="./src/llama_stack_client/resources/agents/agents.py">create</a>(\*\*<a href="src/llama_stack_client/types/agent_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agent_create_response.py">AgentCreateResponse</a></code>
- <code title="post /agents/delete">client.agents.<a href="./src/llama_stack_client/resources/agents/agents.py">delete</a>(\*\*<a href="src/llama_stack_client/types/agent_delete_params.py">params</a>) -> None</code>

### Sessions

Types:

```python
from llama_stack_client.types.agents import Session, SessionCreateResponse
```

Methods:

- <code title="post /agents/session/create">client.agents.sessions.<a href="./src/llama_stack_client/resources/agents/sessions.py">create</a>(\*\*<a href="src/llama_stack_client/types/agents/session_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agents/session_create_response.py">SessionCreateResponse</a></code>
- <code title="post /agents/session/get">client.agents.sessions.<a href="./src/llama_stack_client/resources/agents/sessions.py">retrieve</a>(\*\*<a href="src/llama_stack_client/types/agents/session_retrieve_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agents/session.py">Session</a></code>
- <code title="post /agents/session/delete">client.agents.sessions.<a href="./src/llama_stack_client/resources/agents/sessions.py">delete</a>(\*\*<a href="src/llama_stack_client/types/agents/session_delete_params.py">params</a>) -> None</code>

### Steps

Types:

```python
from llama_stack_client.types.agents import AgentsStep
```

Methods:

- <code title="get /agents/step/get">client.agents.steps.<a href="./src/llama_stack_client/resources/agents/steps.py">retrieve</a>(\*\*<a href="src/llama_stack_client/types/agents/step_retrieve_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agents/agents_step.py">AgentsStep</a></code>

### Turns

Types:

```python
from llama_stack_client.types.agents import AgentsTurnStreamChunk, Turn, TurnStreamEvent
```

Methods:

- <code title="post /agents/turn/create">client.agents.turns.<a href="./src/llama_stack_client/resources/agents/turns.py">create</a>(\*\*<a href="src/llama_stack_client/types/agents/turn_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agents/agents_turn_stream_chunk.py">AgentsTurnStreamChunk</a></code>
- <code title="get /agents/turn/get">client.agents.turns.<a href="./src/llama_stack_client/resources/agents/turns.py">retrieve</a>(\*\*<a href="src/llama_stack_client/types/agents/turn_retrieve_params.py">params</a>) -> <a href="./src/llama_stack_client/types/agents/turn.py">Turn</a></code>

## Datasets

Types:

```python
from llama_stack_client.types import TrainEvalDataset
```

Methods:

- <code title="post /datasets/create">client.datasets.<a href="./src/llama_stack_client/resources/datasets.py">create</a>(\*\*<a href="src/llama_stack_client/types/dataset_create_params.py">params</a>) -> None</code>
- <code title="post /datasets/delete">client.datasets.<a href="./src/llama_stack_client/resources/datasets.py">delete</a>(\*\*<a href="src/llama_stack_client/types/dataset_delete_params.py">params</a>) -> None</code>
- <code title="get /datasets/get">client.datasets.<a href="./src/llama_stack_client/resources/datasets.py">get</a>(\*\*<a href="src/llama_stack_client/types/dataset_get_params.py">params</a>) -> <a href="./src/llama_stack_client/types/train_eval_dataset.py">TrainEvalDataset</a></code>

## Evaluate

Types:

```python
from llama_stack_client.types import EvaluationJob
```

### Jobs

Types:

```python
from llama_stack_client.types.evaluate import (
    EvaluationJobArtifacts,
    EvaluationJobLogStream,
    EvaluationJobStatus,
)
```

Methods:

- <code title="get /evaluate/jobs">client.evaluate.jobs.<a href="./src/llama_stack_client/resources/evaluate/jobs/jobs.py">list</a>() -> <a href="./src/llama_stack_client/types/evaluation_job.py">EvaluationJob</a></code>
- <code title="post /evaluate/job/cancel">client.evaluate.jobs.<a href="./src/llama_stack_client/resources/evaluate/jobs/jobs.py">cancel</a>(\*\*<a href="src/llama_stack_client/types/evaluate/job_cancel_params.py">params</a>) -> None</code>

#### Artifacts

Methods:

- <code title="get /evaluate/job/artifacts">client.evaluate.jobs.artifacts.<a href="./src/llama_stack_client/resources/evaluate/jobs/artifacts.py">list</a>(\*\*<a href="src/llama_stack_client/types/evaluate/jobs/artifact_list_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluate/evaluation_job_artifacts.py">EvaluationJobArtifacts</a></code>

#### Logs

Methods:

- <code title="get /evaluate/job/logs">client.evaluate.jobs.logs.<a href="./src/llama_stack_client/resources/evaluate/jobs/logs.py">list</a>(\*\*<a href="src/llama_stack_client/types/evaluate/jobs/log_list_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluate/evaluation_job_log_stream.py">EvaluationJobLogStream</a></code>

#### Status

Methods:

- <code title="get /evaluate/job/status">client.evaluate.jobs.status.<a href="./src/llama_stack_client/resources/evaluate/jobs/status.py">list</a>(\*\*<a href="src/llama_stack_client/types/evaluate/jobs/status_list_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluate/evaluation_job_status.py">EvaluationJobStatus</a></code>

### QuestionAnswering

Methods:

- <code title="post /evaluate/question_answering/">client.evaluate.question_answering.<a href="./src/llama_stack_client/resources/evaluate/question_answering.py">create</a>(\*\*<a href="src/llama_stack_client/types/evaluate/question_answering_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluation_job.py">EvaluationJob</a></code>

## Evaluations

Methods:

- <code title="post /evaluate/summarization/">client.evaluations.<a href="./src/llama_stack_client/resources/evaluations.py">summarization</a>(\*\*<a href="src/llama_stack_client/types/evaluation_summarization_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluation_job.py">EvaluationJob</a></code>
- <code title="post /evaluate/text_generation/">client.evaluations.<a href="./src/llama_stack_client/resources/evaluations.py">text_generation</a>(\*\*<a href="src/llama_stack_client/types/evaluation_text_generation_params.py">params</a>) -> <a href="./src/llama_stack_client/types/evaluation_job.py">EvaluationJob</a></code>

## Inference

Types:

```python
from llama_stack_client.types import (
    ChatCompletionStreamChunk,
    CompletionStreamChunk,
    TokenLogProbs,
    InferenceChatCompletionResponse,
    InferenceCompletionResponse,
)
```

Methods:

- <code title="post /inference/chat_completion">client.inference.<a href="./src/llama_stack_client/resources/inference/inference.py">chat_completion</a>(\*\*<a href="src/llama_stack_client/types/inference_chat_completion_params.py">params</a>) -> <a href="./src/llama_stack_client/types/inference_chat_completion_response.py">InferenceChatCompletionResponse</a></code>
- <code title="post /inference/completion">client.inference.<a href="./src/llama_stack_client/resources/inference/inference.py">completion</a>(\*\*<a href="src/llama_stack_client/types/inference_completion_params.py">params</a>) -> <a href="./src/llama_stack_client/types/inference_completion_response.py">InferenceCompletionResponse</a></code>

### Embeddings

Types:

```python
from llama_stack_client.types.inference import Embeddings
```

Methods:

- <code title="post /inference/embeddings">client.inference.embeddings.<a href="./src/llama_stack_client/resources/inference/embeddings.py">create</a>(\*\*<a href="src/llama_stack_client/types/inference/embedding_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/inference/embeddings.py">Embeddings</a></code>

## Safety

Types:

```python
from llama_stack_client.types import RunSheidResponse
```

Methods:

- <code title="post /safety/run_shield">client.safety.<a href="./src/llama_stack_client/resources/safety.py">run_shield</a>(\*\*<a href="src/llama_stack_client/types/safety_run_shield_params.py">params</a>) -> <a href="./src/llama_stack_client/types/run_sheid_response.py">RunSheidResponse</a></code>

## Memory

Types:

```python
from llama_stack_client.types import (
    QueryDocuments,
    MemoryCreateResponse,
    MemoryRetrieveResponse,
    MemoryListResponse,
    MemoryDropResponse,
)
```

Methods:

- <code title="post /memory/create">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">create</a>(\*\*<a href="src/llama_stack_client/types/memory_create_params.py">params</a>) -> <a href="./src/llama_stack_client/types/memory_create_response.py">object</a></code>
- <code title="get /memory/get">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">retrieve</a>(\*\*<a href="src/llama_stack_client/types/memory_retrieve_params.py">params</a>) -> <a href="./src/llama_stack_client/types/memory_retrieve_response.py">object</a></code>
- <code title="post /memory/update">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">update</a>(\*\*<a href="src/llama_stack_client/types/memory_update_params.py">params</a>) -> None</code>
- <code title="get /memory/list">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">list</a>() -> <a href="./src/llama_stack_client/types/memory_list_response.py">object</a></code>
- <code title="post /memory/drop">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">drop</a>(\*\*<a href="src/llama_stack_client/types/memory_drop_params.py">params</a>) -> str</code>
- <code title="post /memory/insert">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">insert</a>(\*\*<a href="src/llama_stack_client/types/memory_insert_params.py">params</a>) -> None</code>
- <code title="post /memory/query">client.memory.<a href="./src/llama_stack_client/resources/memory/memory.py">query</a>(\*\*<a href="src/llama_stack_client/types/memory_query_params.py">params</a>) -> <a href="./src/llama_stack_client/types/query_documents.py">QueryDocuments</a></code>

### Documents

Types:

```python
from llama_stack_client.types.memory import DocumentRetrieveResponse
```

Methods:

- <code title="post /memory/documents/get">client.memory.documents.<a href="./src/llama_stack_client/resources/memory/documents.py">retrieve</a>(\*\*<a href="src/llama_stack_client/types/memory/document_retrieve_params.py">params</a>) -> <a href="./src/llama_stack_client/types/memory/document_retrieve_response.py">DocumentRetrieveResponse</a></code>
- <code title="post /memory/documents/delete">client.memory.documents.<a href="./src/llama_stack_client/resources/memory/documents.py">delete</a>(\*\*<a href="src/llama_stack_client/types/memory/document_delete_params.py">params</a>) -> None</code>

## PostTraining

Types:

```python
from llama_stack_client.types import PostTrainingJob
```

Methods:

- <code title="post /post_training/preference_optimize">client.post_training.<a href="./src/llama_stack_client/resources/post_training/post_training.py">preference_optimize</a>(\*\*<a href="src/llama_stack_client/types/post_training_preference_optimize_params.py">params</a>) -> <a href="./src/llama_stack_client/types/post_training_job.py">PostTrainingJob</a></code>
- <code title="post /post_training/supervised_fine_tune">client.post_training.<a href="./src/llama_stack_client/resources/post_training/post_training.py">supervised_fine_tune</a>(\*\*<a href="src/llama_stack_client/types/post_training_supervised_fine_tune_params.py">params</a>) -> <a href="./src/llama_stack_client/types/post_training_job.py">PostTrainingJob</a></code>

### Jobs

Types:

```python
from llama_stack_client.types.post_training import (
    PostTrainingJobArtifacts,
    PostTrainingJobLogStream,
    PostTrainingJobStatus,
)
```

Methods:

- <code title="get /post_training/jobs">client.post_training.jobs.<a href="./src/llama_stack_client/resources/post_training/jobs.py">list</a>() -> <a href="./src/llama_stack_client/types/post_training_job.py">PostTrainingJob</a></code>
- <code title="get /post_training/job/artifacts">client.post_training.jobs.<a href="./src/llama_stack_client/resources/post_training/jobs.py">artifacts</a>(\*\*<a href="src/llama_stack_client/types/post_training/job_artifacts_params.py">params</a>) -> <a href="./src/llama_stack_client/types/post_training/post_training_job_artifacts.py">PostTrainingJobArtifacts</a></code>
- <code title="post /post_training/job/cancel">client.post_training.jobs.<a href="./src/llama_stack_client/resources/post_training/jobs.py">cancel</a>(\*\*<a href="src/llama_stack_client/types/post_training/job_cancel_params.py">params</a>) -> None</code>
- <code title="get /post_training/job/logs">client.post_training.jobs.<a href="./src/llama_stack_client/resources/post_training/jobs.py">logs</a>(\*\*<a href="src/llama_stack_client/types/post_training/job_logs_params.py">params</a>) -> <a href="./src/llama_stack_client/types/post_training/post_training_job_log_stream.py">PostTrainingJobLogStream</a></code>
- <code title="get /post_training/job/status">client.post_training.jobs.<a href="./src/llama_stack_client/resources/post_training/jobs.py">status</a>(\*\*<a href="src/llama_stack_client/types/post_training/job_status_params.py">params</a>) -> <a href="./src/llama_stack_client/types/post_training/post_training_job_status.py">PostTrainingJobStatus</a></code>

## RewardScoring

Types:

```python
from llama_stack_client.types import RewardScoring, ScoredDialogGenerations
```

Methods:

- <code title="post /reward_scoring/score">client.reward_scoring.<a href="./src/llama_stack_client/resources/reward_scoring.py">score</a>(\*\*<a href="src/llama_stack_client/types/reward_scoring_score_params.py">params</a>) -> <a href="./src/llama_stack_client/types/reward_scoring.py">RewardScoring</a></code>

## SyntheticDataGeneration

Types:

```python
from llama_stack_client.types import SyntheticDataGeneration
```

Methods:

- <code title="post /synthetic_data_generation/generate">client.synthetic_data_generation.<a href="./src/llama_stack_client/resources/synthetic_data_generation.py">generate</a>(\*\*<a href="src/llama_stack_client/types/synthetic_data_generation_generate_params.py">params</a>) -> <a href="./src/llama_stack_client/types/synthetic_data_generation.py">SyntheticDataGeneration</a></code>

## BatchInference

Types:

```python
from llama_stack_client.types import BatchChatCompletion
```

Methods:

- <code title="post /batch_inference/chat_completion">client.batch_inference.<a href="./src/llama_stack_client/resources/batch_inference.py">chat_completion</a>(\*\*<a href="src/llama_stack_client/types/batch_inference_chat_completion_params.py">params</a>) -> <a href="./src/llama_stack_client/types/batch_chat_completion.py">BatchChatCompletion</a></code>
- <code title="post /batch_inference/completion">client.batch_inference.<a href="./src/llama_stack_client/resources/batch_inference.py">completion</a>(\*\*<a href="src/llama_stack_client/types/batch_inference_completion_params.py">params</a>) -> <a href="./src/llama_stack_client/types/shared/batch_completion.py">BatchCompletion</a></code>

## Models

Types:

```python
from llama_stack_client.types import ModelServingSpec
```

Methods:

- <code title="get /models/list">client.models.<a href="./src/llama_stack_client/resources/models.py">list</a>() -> <a href="./src/llama_stack_client/types/model_serving_spec.py">ModelServingSpec</a></code>
- <code title="get /models/get">client.models.<a href="./src/llama_stack_client/resources/models.py">get</a>(\*\*<a href="src/llama_stack_client/types/model_get_params.py">params</a>) -> <a href="./src/llama_stack_client/types/model_serving_spec.py">Optional</a></code>

## MemoryBanks

Types:

```python
from llama_stack_client.types import MemoryBankSpec
```

Methods:

- <code title="get /memory_banks/list">client.memory_banks.<a href="./src/llama_stack_client/resources/memory_banks.py">list</a>() -> <a href="./src/llama_stack_client/types/memory_bank_spec.py">MemoryBankSpec</a></code>
- <code title="get /memory_banks/get">client.memory_banks.<a href="./src/llama_stack_client/resources/memory_banks.py">get</a>(\*\*<a href="src/llama_stack_client/types/memory_bank_get_params.py">params</a>) -> <a href="./src/llama_stack_client/types/memory_bank_spec.py">Optional</a></code>

## Shields

Types:

```python
from llama_stack_client.types import ShieldSpec
```

Methods:

- <code title="get /shields/list">client.shields.<a href="./src/llama_stack_client/resources/shields.py">list</a>() -> <a href="./src/llama_stack_client/types/shield_spec.py">ShieldSpec</a></code>
- <code title="get /shields/get">client.shields.<a href="./src/llama_stack_client/resources/shields.py">get</a>(\*\*<a href="src/llama_stack_client/types/shield_get_params.py">params</a>) -> <a href="./src/llama_stack_client/types/shield_spec.py">Optional</a></code>
