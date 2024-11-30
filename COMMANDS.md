```bash

# Using Conda now
python -m venv .venv
source $STORAGE_DIR/llama-stack/.venv/bin/activate

source ~/miniconda3/bin/activate
conda create --prefix ./envs python=3.10 

source ~/miniconda3/bin/activate
conda activate ./envs

pip install -e .

huggingface-cli login

export $(cat .env | xargs)

# Env vars:
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export LLAMA_STACK_PORT=5001
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export INFERENCE_PORT=8000
export VLLM_URL=http://localhost:8000/v1
export SQLITE_STORE_DIR=$LLAMA_STACK_CONFIG_DIR/distributions/meta-reference-gpu
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

# vLLM server
export $(cat .env | xargs)
sudo docker run --gpus all \
    -v $STORAGE_DIR/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)" \
    -p 8000:$INFERENCE_PORT \
    --ipc=host \
    --net=host \
    vllm/vllm-openai:v0.6.3.post1 \
    --model $INFERENCE_MODEL

# Remote vLLM
export $(cat .env | xargs)
sudo docker run \
  -it \
  --net=host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-remote-vllm:0.0.54 \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://localhost:$INFERENCE_PORT/v1

llama model download --model-id meta-llama/Llama-3.2-3B-Instruct
# Add in signed URL from email

# Meta reference gpu server
export $(cat .env | xargs)
sudo docker run \
  -it \
  -v ~/.llama:/root/.llama \
  --gpus all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-meta-reference-gpu \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Fireworks server
sudo docker run \
    -it \
    -v ~/run.yaml:/root/run.yaml \
    --net=host \
    llamastack/distribution-fireworks \
    --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
    --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY



llama-stack-client --endpoint http://localhost:$LLAMA_STACK_PORT   inference chat-completion   --message "hello, what model are you?"



# Install the stack
llama stack build --template remote-vllm --image-type conda
# Run the stack
conda activate llamastack-remote-vllm
llama stack run run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

llama stack build --template meta-reference-gpu --image-type conda && llama stack run distributions/meta-reference-gpu/run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

llama stack build --template meta-reference-gpu --image-type conda && llama stack run distributions/meta-reference-gpu/run-with-safety.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct

llama download --model-id Llama3.2-11B-Vision-Instruct
llama download --model-id Llama3.2-3B-Instruct
llama download --model-id Llama-Guard-3-1B

ls $SQLITE_STORE_DIR
sudo apt install sqlite3
# Faiss store
sqlite3 $SQLITE_STORE_DIR/faiss_store.db
.tables
.schema
.headers ON
.mode column
.output sql.txt
select key from kvstore;
select * from kvstore where key = 'memory_banks:v1::test_bank_2';
.output sql.txt;
select * from kvstore where key = 'faiss_index:v1::test_bank_2';

# Registry
sqlite3 $SQLITE_STORE_DIR/registry.db
select key from kvstore;
select * from kvstore where key = 'distributions:registry:v2::model:meta-llama/Llama-3.2-11B-Vision-Instruct';

# Agent store
sqlite3 $SQLITE_STORE_DIR/agents_store.db
select key from kvstore;
# Session
select * from kvstore where key = 'session:f4920b89-1035-4432-92ab-3d800878e28d:7b19e203-53cc-4295-b6cf-f0c400611ed1';
# Turns
.output sql.txt
select * from kvstore where key = 'session:f4920b89-1035-4432-92ab-3d800878e28d:7b19e203-53cc-4295-b6cf-f0c400611ed1:e38da75e-70fb-4895-b522-b25373f3e8d5';
# Agents
select * from kvstore where key = 'agent:f4920b89-1035-4432-92ab-3d800878e28d';


conda create --prefix ./faiss-env python=3.10


source ~/miniconda3/bin/activate
conda activate ./faiss-env

pip install "numpy<2.0" faiss-gpu aiosqlite sentence-transformers
python inspect_faiss.py





```
