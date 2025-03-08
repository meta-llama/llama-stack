## Safety Guardrails

Safety is a critical component of any AI application. Llama Stack provides a Shield system that can be applied at multiple touchpoints:

```python
from llama_stack_client import LlamaStackClient

# Replace host and port
client = LlamaStackClient(base_url=f"http://{HOST}:{PORT}")

# Register a safety shield
shield_id = "content_safety"
SHEILD_NAME = "meta-llama/Llama-Guard-3-1B"
# If no provider specified and multiple providers available, need specify provider_id, e.g. provider_id="llama-guard"
# Check with `llama model list` for the supported Llama Guard type models
client.shields.register(shield_id=shield_id, provider_shield_id=SHEILD_NAME)

# Run content through shield
# To trigger a violation result, try inputting some sensitive content in <User message here>
response = client.safety.run_shield(
    shield_id=SHEILD_NAME,
    messages=[{"role": "user", "content": "User message here"}],
    params={},
)

if response.violation:
    print(f"Safety violation detected: {response.violation.user_message}")
else:
    print("The input content does not trigger any violations.")
```
