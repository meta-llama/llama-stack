FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y        iputils-ping net-tools iproute2 dnsutils telnet        curl wget telnet        procps psmisc lsof        traceroute        bubblewrap        && rm -rf /var/lib/apt/lists/*

RUN pip install /app/llama-stack-source
RUN pip uninstall -y llama-models
RUN pip install /app/llama-models-source

RUN pip install fastapi fire httpx uvicorn accelerate blobfile fairscale fbgemm-gpu==0.8.0 torch torchvision transformers zmq codeshield transformers matplotlib pillow pandas scikit-learn aiosqlite psycopg2-binary redis blobfile chardet pypdf tqdm numpy scikit-learn scipy nltk sentencepiece transformers faiss-cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers --no-deps

# This would be good in production but for debugging flexibility lets not add it right now
# We need a more solid production ready entrypoint.sh anyway
#
CMD ["python", "-m", "llama_stack.distribution.server.server", "./llamastack-run.yaml"]

ADD tmp/configs/local-gpu-build.yaml ./llamastack-build.yaml
ADD tmp/configs/local-gpu-run.yaml ./llamastack-run.yaml
