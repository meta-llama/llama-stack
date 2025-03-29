## Telemetry

The Llama Stack telemetry system provides comprehensive tracing, metrics, and logging capabilities. It supports multiple sink types including OpenTelemetry, SQLite, and Console output.

### Events
The telemetry system supports three main types of events:

- **Unstructured Log Events**: Free-form log messages with severity levels
```python
unstructured_log_event = UnstructuredLogEvent(
    message="This is a log message", severity=LogSeverity.INFO
)
```
- **Metric Events**: Numerical measurements with units
```python
metric_event = MetricEvent(metric="my_metric", value=10, unit="count")
```
- **Structured Log Events**: System events like span start/end. Extensible to add more structured log types.
```python
structured_log_event = SpanStartPayload(name="my_span", parent_span_id="parent_span_id")
```

### Spans and Traces
- **Spans**: Represent operations with timing and hierarchical relationships
- **Traces**: Collection of related spans forming a complete request flow

### Sinks
- **OpenTelemetry**: Send events to an OpenTelemetry Collector. This is useful for visualizing traces in a tool like Jaeger.
- **SQLite**: Store events in a local SQLite database. This is needed if you want to query the events later through the Llama Stack API.
- **Console**: Print events to the console.

### Providers

#### Meta-Reference Provider
Currently, only the meta-reference provider is implemented. It can be configured to send events to three sink types:
1) OpenTelemetry Collector
2) SQLite
3) Console

#### Configuration

Here's an example that sends telemetry signals to all three sink types. Your configuration might use only one.
```yaml
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      sinks: ['console', 'sqlite', 'otel_trace', 'otel_metric']
      otel_trace_endpoint: "http://localhost:4318/v1/traces"
      otel_metric_endpoint: "http://localhost:4318/v1/metrics"
      sqlite_db_path: "/path/to/telemetry.db"
```

### Jaeger to visualize traces

The `otel` sink works with any service compatible with the OpenTelemetry collector, traces and metrics has two separate endpoints.
Let's use Jaeger to visualize this data.

Start a Jaeger instance with the OTLP HTTP endpoint at 4318 and the Jaeger UI at 16686 using the following command:

```bash
$ docker run --pull always --rm --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/jaeger:2.1.0
```

Once the Jaeger instance is running, you can visualize traces by navigating to http://localhost:16686/.

### Querying Traces Stored in SQLite

The `sqlite` sink allows you to query traces without an external system. Here are some example queries. Refer to the notebook at [Llama Stack Building AI Applications](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb) for more examples on how to query traces and spaces.
