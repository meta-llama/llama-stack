# Using Llama Stack as Library
```
# You can use distribution template names e.g. "ollama", "together", "vllm"
LLAMA_STACK_CONFIG=ollama pytest tests/client-sdk/memory -v
```

# Using local Llama Stack server instance
```
# Export Llama Stack naming vars
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# Export Ollama naming vars
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export OLLAMA_SAFETY_MODEL="llama-guard3:1b"

# Start Ollama instance
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
ollama run $OLLAMA_SAFETY_MODEL --keepalive 60m

# Start the Llama Stack server
llama stack run ./llama_stack/templates/ollama/run.yaml

# Run the tests
LLAMA_STACK_BASE_URL=http://localhost:5000 pytest tests/client-sdk/memory -v
```
