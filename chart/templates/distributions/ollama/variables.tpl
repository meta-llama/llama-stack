{{- define "distributions.ollama" -}}
image:
  repository: ollama/ollama
  tag: 0.5.8
models:
  "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b-instruct-fp16"
{{- end -}} 
