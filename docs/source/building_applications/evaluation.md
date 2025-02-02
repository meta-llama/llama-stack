## Testing & Evaluation

Llama Stack provides built-in tools for evaluating your applications:

1. **Benchmarking**: Test against standard datasets
2. **Application Evaluation**: Score your application's outputs
3. **Custom Metrics**: Define your own evaluation criteria

Here's how to set up basic evaluation:

```python
# Create an evaluation task
response = client.eval_tasks.register(
    eval_task_id="my_eval",
    dataset_id="my_dataset",
    scoring_functions=["accuracy", "relevance"],
)

# Run evaluation
job = client.eval.run_eval(
    task_id="my_eval",
    task_config={
        "type": "app",
        "eval_candidate": {"type": "agent", "config": agent_config},
    },
)

# Get results
result = client.eval.job_result(task_id="my_eval", job_id=job.job_id)
```
