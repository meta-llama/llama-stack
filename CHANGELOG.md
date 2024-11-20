# Changelog

## 0.0.53

### Added
- Resource-oriented design for models, shields, memory banks, datasets and eval tasks
- Persistence for registered objects with distribution
- Ability to persist memory banks created for FAISS
- PostgreSQL KVStore implementation
- Environment variable placeholder support in run.yaml files
- Provider deprecation support
- Comprehensive Zero-to-Hero notebooks and quickstart guides
- Colab notebook integration for Llama Stack lesson
- Remote::vllm provider with vision model support
- Support for quantized models in Ollama
- Vision models support for Together, Fireworks, Meta-Reference, and Ollama
- Bedrock distribution with safety shields support

### Changed
- Split safety into distinct providers (llama-guard, prompt-guard, code-scanner)
- Improved distribution registry with SQLite default storage
- Enhanced test infrastructure with composable fixtures
- Changed provider naming convention (`impls` → `inline`, `adapters` → `remote`)
- Updated API signatures for dataset and eval task registration
- Restructured folder organization for providers
- Enhanced Docker build configuration
- Added version prefixing for REST API routes

### Removed
- `llama stack configure` command

### Fixed
- Agent system instructions persistence after first turn
- PostgreSQL config validation
- vLLM adapter chat completion signature
- Routing table lookup key for memory banks
- Together inference validator
- Exception handling for SSE connection closure

### Development
- Added new test migration system
- Enhanced test setup with configurable model selection
- Added support for remote providers in tests
- Improved pre-commit hooks configuration
- Updated OpenAPI generator and specifications
