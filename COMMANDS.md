```bash
source ~/miniconda3/bin/activate
conda create --prefix ./envs python=3.10

source ~/miniconda3/bin/activate
conda activate ./envs

pip install -e . \
&& llama stack build --config ./build.yaml --image-type conda \
&& llama stack run ./run.yaml \
  --port 5001

pytest llama_stack/providers/tests/inference/test_text_inference.py -v -k groq --lf -s

trash .git/hooks/pre-commit

llama stack build --template ollama --image-type conda \
&& llama stack run ./distributions/ollama/run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```
