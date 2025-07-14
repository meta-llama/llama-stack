# remote::opengauss

## Description


[OpenGauss](https://opengauss.org/en/) is a remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with Llama Stack

## Usage

To use OpenGauss in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use OpenGauss.
3. Start storing and querying vectors.

## Installation

You can install OpenGauss using docker:

```bash
docker pull opengauss/opengauss:latest
```
## Documentation
See [OpenGauss' documentation](https://docs.opengauss.org/en/docs/5.0.0/docs/GettingStarted/understanding-opengauss.html) for more details about OpenGauss in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | `str \| None` | No | localhost |  |
| `port` | `int \| None` | No | 5432 |  |
| `db` | `str \| None` | No | postgres |  |
| `user` | `str \| None` | No | postgres |  |
| `password` | `str \| None` | No | mysecretpassword |  |

## Sample Configuration

```yaml
host: ${env.OPENGAUSS_HOST:=localhost}
port: ${env.OPENGAUSS_PORT:=5432}
db: ${env.OPENGAUSS_DB}
user: ${env.OPENGAUSS_USER}
password: ${env.OPENGAUSS_PASSWORD}

```

