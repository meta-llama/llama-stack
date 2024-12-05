# Telemetry
```{note}
The telemetry system is currently experimental and subject to change. We welcome feedback and contributions to help improve it.
```



The Llama Stack telemetry system provides comprehensive tracing, metrics, and logging capabilities. It supports multiple sink types including OpenTelemetry, SQLite, and Console output.

## Key Concepts

### Events
The telemetry system supports three main types of events:

- **Unstructured Log Events**: Free-form log messages with severity levels
- **Metric Events**: Numerical measurements with units
- **Structured Log Events**: System events like span start/end

### Spans and Traces
- **Spans**: Represent operations with timing and hierarchical relationships
- **Traces**: Collection of related spans forming a complete request flow

### Sinks
- **OpenTelemetry**: Send events to an OpenTelemetry Collector. This is useful for visualizing traces in a service like Jaeger.
- **SQLite**: Store events in a local SQLite database. This is needed if you want to query the events later through the Llama Stack API.
- **Console**: Print events to the console.


## Providers

### Meta-Reference Provider
Currently, only the meta-reference provider is implemented. It can be configured to send events to three sink types:
1) OpenTelemetry Collector
2) SQLite
3) Console

## Configuration

```yaml
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      sinks: ['console', 'sqlite', 'otel']
      otel_endpoint: "http://localhost:4318/v1/traces"
      sqlite_db_path: "/path/to/telemetry.db"
```


## Querying Examples

Querying Traces for a agent session

``` bash
 curl -X POST 'http://localhost:5000/alpha/telemetry/query-traces' \
-H 'Content-Type: application/json' \
-d '{
  "attribute_filters": [
    {
      "key": "session_id",
      "op": "eq",
      "value": "dd667b87-ca4b-4d30-9265-5a0de318fc65" }],
  "limit": 100,
  "offset": 0,
  "order_by": ["start_time"]

  [
  {
    "trace_id": "6902f54b83b4b48be18a6f422b13e16f",
    "root_span_id": "5f37b85543afc15a",
    "start_time": "2024-12-04T08:08:30.501587",
    "end_time": "2024-12-04T08:08:36.026463"
  },
  ........
]
}'

```

Querying spans for a specifc root span id

``` bash
curl -X POST 'http://localhost:5000/alpha/telemetry/get-span-tree' \
-H 'Content-Type: application/json' \
-d '{ "span_id" : "6cceb4b48a156913", "max_depth": 2 }'

{
  "span_id": "6cceb4b48a156913",
  "trace_id": "dafa796f6aaf925f511c04cd7c67fdda",
  "parent_span_id": "892a66d726c7f990",
  "name": "retrieve_rag_context",
  "start_time": "2024-12-04T09:28:21.781995",
  "end_time": "2024-12-04T09:28:21.913352",
  "attributes": {
    "input": [
      "{\"role\":\"system\",\"content\":\"You are a helpful assistant\"}",
      "{\"role\":\"user\",\"content\":\"What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.\",\"context\":null}"
    ]
  },
  "children": [
    {
      "span_id": "1a2df181854064a8",
      "trace_id": "dafa796f6aaf925f511c04cd7c67fdda",
      "parent_span_id": "6cceb4b48a156913",
      "name": "MemoryRouter.query_documents",
      "start_time": "2024-12-04T09:28:21.787620",
      "end_time": "2024-12-04T09:28:21.906512",
      "attributes": {
        "input": null
      },
      "children": [],
      "status": "ok"
    }
  ],
  "status": "ok"
}

```
