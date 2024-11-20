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
- Support for quantized models in Ollama
- Vision models support for Together, Fireworks, Meta-Reference, and Ollama, and vLLM
- Bedrock distribution with safety shields support
- Evals API with task registration and scoring functions
- MMLU and SimpleQA benchmark scoring functions
- Huggingface dataset provider integration for benchmarks
- Support for custom dataset registration from local paths
- Benchmark evaluation CLI tools with visualization tables
- RAG evaluation scoring functions and metrics
- Local persistence for datasets and eval tasks

### Changed
- Split safety into distinct providers (llama-guard, prompt-guard, code-scanner)
- Changed provider naming convention (`impls` → `inline`, `adapters` → `remote`)
- Updated API signatures for dataset and eval task registration
- Restructured folder organization for providers
- Enhanced Docker build configuration
- Added version prefixing for REST API routes
- Enhanced evaluation task registration workflow
- Improved benchmark evaluation output formatting
- Restructured evals folder organization for better modularity

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
