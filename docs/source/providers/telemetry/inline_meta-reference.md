# inline::meta-reference

## Description

Meta's reference implementation of telemetry and observability using OpenTelemetry.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `otel_trace_endpoint` | `str \| None` | No |  | The OpenTelemetry collector endpoint URL for traces |
| `otel_metric_endpoint` | `str \| None` | No |  | The OpenTelemetry collector endpoint URL for metrics |
| `service_name` | `<class 'str'>` | No | ​ | The service name to use for telemetry |
| `sinks` | `list[inline.telemetry.meta_reference.config.TelemetrySink` | No | [<TelemetrySink.CONSOLE: 'console'>, <TelemetrySink.SQLITE: 'sqlite'>] | List of telemetry sinks to enable (possible values: otel, sqlite, console) |
| `sqlite_db_path` | `<class 'str'>` | No | ~/.llama/runtime/trace_store.db | The path to the SQLite database to use for storing traces |

## Sample Configuration

```yaml
service_name: "${env.OTEL_SERVICE_NAME:=\u200B}"
sinks: ${env.TELEMETRY_SINKS:=console,sqlite}
sqlite_db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/trace_store.db

```

