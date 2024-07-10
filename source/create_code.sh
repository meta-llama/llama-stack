#!/bin/bash 

set -euo pipefail 
set -x 

export JAVA_HOME=/usr/local/java-runtime/impl/11

$JAVA_HOME/bin/java -jar codegen/openapi-generator-cli.jar \
  generate \
  -i openapi.yaml \
  -g python-flask \
  -o /tmp/foo \
  --log-to-stderr \
  --global-property debugModels,debugOperations,debugOpenAPI,debugSupportingFiles
