The RFC Specification (OpenAPI format) is generated from the set of API endpoints located in `llama_stack/[<subdir>]/api/endpoints.py` using the `generate.py` utility.

Please install the following packages before running the script:

```
pip install python-openapi json-strong-typing fire PyYAML llama-models
```

Then simply run `sh run_openapi_generator.sh <OUTPUT_DIR>`
