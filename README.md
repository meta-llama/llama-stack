This repo contains the API specifications for various parts of the Llama Stack.
The Stack consists of toolchain-apis and agentic-apis. 

The tool chain apis that are covered -- 
- inference / batch inference
- post training
- reward model scoring
- synthetic data generation


### Generate OpenAPI specs 

Set up virtual environment 

```
python3.9 -m venv ~/.venv/toolchain/ 
source ~/.venv/toolchain/bin/activate

with-proxy pip3 install -r requirements.txt 

```

Run the generate.sh script 

```
cd source && sh generate.sh
```
