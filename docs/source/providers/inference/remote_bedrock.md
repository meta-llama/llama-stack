# remote::bedrock

## Description

AWS Bedrock inference provider for accessing various AI models through AWS's managed service.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `aws_access_key_id` | `str \| None` | No |  | The AWS access key to use. Default use environment variable: AWS_ACCESS_KEY_ID |
| `aws_secret_access_key` | `pydantic.types.SecretStr \| None` | No |  | The AWS secret access key to use. Default use environment variable: AWS_SECRET_ACCESS_KEY |
| `aws_session_token` | `pydantic.types.SecretStr \| None` | No |  | The AWS session token to use. Default use environment variable: AWS_SESSION_TOKEN |
| `region_name` | `str \| None` | No |  | The default AWS Region to use, for example, us-west-1 or us-west-2.Default use environment variable: AWS_DEFAULT_REGION |
| `profile_name` | `str \| None` | No |  | The profile name that contains credentials to use.Default use environment variable: AWS_PROFILE |
| `total_max_attempts` | `int \| None` | No |  | An integer representing the maximum number of attempts that will be made for a single request, including the initial attempt. Default use environment variable: AWS_MAX_ATTEMPTS |
| `retry_mode` | `str \| None` | No |  | A string representing the type of retries Boto3 will perform.Default use environment variable: AWS_RETRY_MODE |
| `connect_timeout` | `float \| None` | No | 60 | The time in seconds till a timeout exception is thrown when attempting to make a connection. The default is 60 seconds. |
| `read_timeout` | `float \| None` | No | 60 | The time in seconds till a timeout exception is thrown when attempting to read from a connection.The default is 60 seconds. |
| `session_ttl` | `int \| None` | No | 3600 | The time in seconds till a session expires. The default is 3600 seconds (1 hour). |

## Sample Configuration

```yaml
{}

```

