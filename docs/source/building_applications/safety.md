## Safety Guardrails

Safety is a critical component of any AI application. Llama Stack provides a Shield system that can be applied at multiple touchpoints:

```python
# Register a safety shield
shield_id = "content_safety"
client.shields.register(shield_id=shield_id, provider_shield_id="llama-guard-basic")

# Run content through shield
response = client.safety.run_shield(
    shield_id=shield_id, messages=[{"role": "user", "content": "User message here"}]
)

if response.violation:
    print(f"Safety violation detected: {response.violation.user_message}")
```
