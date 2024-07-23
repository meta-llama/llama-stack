This repo contains the API specifications for various parts of the Llama Stack.
The Stack consists of toolchain-apis and agentic-apis.

The tool chain apis that are covered --
- inference / batch inference
- post training
- reward model scoring
- synthetic data generation


## Running FP8

You need `fbgemm-gpu` package which requires torch >= 2.4.0 (currently only in nightly, but releasing shortly...).

```bash
ENV=fp8_env
conda create -n $ENV python=3.10
conda activate $ENV

pip3 install -r fp8_requirements.txt
```


### Generate OpenAPI specs

Set up virtual environment

```
python3 -m venv ~/.venv/toolchain/
source ~/.venv/toolchain/bin/activate

with-proxy pip3 install -r requirements.txt

```

Run the generate.sh script

```
cd source && sh generate.sh
```
