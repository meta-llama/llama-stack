{{/* Variables */}}
{{- define "llamaStack" -}}
images:
  ollama:
    repository: llamastack/distribution-ollama
    tag: 0.1.2 # TODO paramaterize this from Chart Version, but requires building images with the same tag scheme
{{- end -}} 

{{/* Validation */}}
{{- if not .Values.distribution -}}
  {{- fail "distribution must be defined" -}}
{{- end -}}
