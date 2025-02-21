{{- $variables := include "distributions.ollama" . | fromYaml -}}
{{- if not (hasKey $variables.models .Values.model) -}}
  {{- fail (printf ".model '%s' is not supported" .Values.model) -}}
{{- end -}}
