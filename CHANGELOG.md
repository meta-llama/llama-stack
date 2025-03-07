# Changelog

# v0.1.5.1
Published on: 2025-02-28T22:37:44Z

## 0.1.5.1 Release Notes
* Fixes for security risk in https://github.com/meta-llama/llama-stack/pull/1327 and https://github.com/meta-llama/llama-stack/pull/1328

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.5...v0.1.5.1

---

# v0.1.5
Published on: 2025-02-28T18:14:01Z

## 0.1.5 Release Notes
###  Build Agents
* Inference: Support more non-llama models (openai, anthropic, gemini)
* Inference: Can use the provider's model name in addition to the HF alias
* Inference: Fixed issues with calling tools that weren't specified in the prompt
* RAG: Improved system prompt for RAG and no more need for hard-coded rag-tool calling
* Embeddings: Added support for Nemo retriever embedding models
* Tools: Added support for MCP tools in Ollama Distribution
* Distributions: Added new Groq distribution

### Customize Models
* Save post-trained checkpoint in SafeTensor format to allow Ollama inference provider to use the post-trained model

### Monitor agents
* More comprehensive logging of agent steps including client tools
* Telemetry inputs/outputs are now structured and queryable
* Ability to retrieve agents session, turn, step by ids

### Better Engineering
* Moved executorch Swift code out of this repo into the llama-stack-client-swift repo, similar to kotlin
* Move most logging to use logger instead of prints
* Completed text /chat-completion and /completion tests

## New Contributors
* @Shreyanand made their first contribution in https://github.com/meta-llama/llama-stack/pull/1283
* @luis5tb made their first contribution in https://github.com/meta-llama/llama-stack/pull/1269

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.4...v0.1.5

---

# v0.1.4
Published on: 2025-02-25T00:02:43Z

## v0.1.4 Release Notes
Here are the key changes coming as part of this release:

### Build and Test Agents
* Inference: Added support for non-llama models
* Inference: Added option to list all downloaded models and remove models
* Agent: Introduce new api agents.resume_turn to include client side tool execution in the same turn
* Agent: AgentConfig introduces new variable “tool_config” that allows for better tool configuration and system prompt overrides
* Agent: Added logging for agent step start and completion times
* Agent: Added support for logging for tool execution metadata
* Embedding: Updated /inference/embeddings to support asymmetric models, truncation and variable sized outputs
* Embedding: Updated embedding models for Ollama, Together, and Fireworks with available defaults
* VectorIO: Improved performance of sqlite-vec using chunked writes
### Agent Evals and Model Customization
* Deprecated api /eval-tasks. Use /eval/benchmark  instead
* Added CPU training support for TorchTune
### Deploy and Monitoring of Agents
* Consistent view of client and server tool calls in telemetry
### Better Engineering
* Made tests more data-driven for consistent evaluation
* Fixed documentation links and improved API reference generation
* Various small fixes for build scripts and system reliability


## New Contributors
* @fulvius31 made their first contribution in https://github.com/meta-llama/llama-stack/pull/1114
* @shrinitg made their first contribution in https://github.com/meta-llama/llama-stack/pull/543
* @raspawar made their first contribution in https://github.com/meta-llama/llama-stack/pull/1174
* @kevincogan made their first contribution in https://github.com/meta-llama/llama-stack/pull/1129
* @LESSuseLESS made their first contribution in https://github.com/meta-llama/llama-stack/pull/1180
* @jland-redhat made their first contribution in https://github.com/meta-llama/llama-stack/pull/1208

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.3...v0.1.4

---

# v0.1.3
Published on: 2025-02-14T20:24:32Z

## v0.1.3 Release

Here are some key changes that are coming as part of this release.

### Build and Test Agents
Streamlined the initial development experience
- Added support for  llama stack run --image-type venv
- Enhanced vector store options with new sqlite-vec provider and improved Qdrant integration
- vLLM improvements for tool calling and logprobs
- Better handling of sporadic code_interpreter tool calls

### Agent Evals
Better benchmarking and Agent performance assessment
- Renamed eval API /eval-task to /benchmarks
- Improved documentation and notebooks for RAG and evals

### Deploy and Monitoring of Agents
Improved production readiness
- Added usage metrics collection for chat completions
- CLI improvements for provider information
- Improved error handling and system reliability
- Better model endpoint handling and accessibility
- Improved signal handling on distro server

### Better Engineering
Infrastructure and code quality improvements
- Faster text-based chat completion tests
- Improved testing for non-streaming agent apis
- Standardized import formatting with ruff linter
- Added conventional commits standard
- Fixed documentation parsing issues

## New Contributors
* @MichaelClifford made their first contribution in https://github.com/meta-llama/llama-stack/pull/1009
* @ellistarn made their first contribution in https://github.com/meta-llama/llama-stack/pull/1035
* @kelbrown20 made their first contribution in https://github.com/meta-llama/llama-stack/pull/992
* @franciscojavierarceo made their first contribution in https://github.com/meta-llama/llama-stack/pull/1040
* @bbrowning made their first contribution in https://github.com/meta-llama/llama-stack/pull/1075
* @reidliu41 made their first contribution in https://github.com/meta-llama/llama-stack/pull/1072
* @vishnoianil made their first contribution in https://github.com/meta-llama/llama-stack/pull/1081

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.2...v0.1.3

---

# v0.1.2
Published on: 2025-02-07T22:06:49Z

# TL;DR
- Several stabilizations to development flows after the switch to `uv`
- Migrated CI workflows to new OSS repo - [llama-stack-ops](https://github.com/meta-llama/llama-stack-ops)
- Added automated rebuilds for ReadTheDocs
- Llama Stack server supports HTTPS
- Added system prompt overrides support
- Several bug fixes and improvements to documentation (check out Kubernetes deployment guide by @terrytangyuan )

## New Contributors
* @nathan-weinberg made their first contribution in https://github.com/meta-llama/llama-stack/pull/939
* @cdoern made their first contribution in https://github.com/meta-llama/llama-stack/pull/954
* @jwm4 made their first contribution in https://github.com/meta-llama/llama-stack/pull/957
* @booxter made their first contribution in https://github.com/meta-llama/llama-stack/pull/961
* @kami619 made their first contribution in https://github.com/meta-llama/llama-stack/pull/960
* @cooktheryan made their first contribution in https://github.com/meta-llama/llama-stack/pull/974
* @aakankshaduggal made their first contribution in https://github.com/meta-llama/llama-stack/pull/976
* @leseb made their first contribution in https://github.com/meta-llama/llama-stack/pull/988
* @mlecanu made their first contribution in https://github.com/meta-llama/llama-stack/pull/997

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.1...v0.1.2

---

# v0.1.1
Published on: 2025-02-02T02:29:24Z

A bunch of small / big improvements everywhere including support for Windows, switching to `uv` and many provider improvements.

## New Contributors
* @BakungaBronson made their first contribution in https://github.com/meta-llama/llama-stack/pull/877
* @Ckhanoyan made their first contribution in https://github.com/meta-llama/llama-stack/pull/888
* @hanzlfs made their first contribution in https://github.com/meta-llama/llama-stack/pull/660
* @dvrogozh made their first contribution in https://github.com/meta-llama/llama-stack/pull/903

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.0...v0.1.1

---

# v0.1.0
Published on: 2025-01-24T17:47:47Z

We are excited to announce a stable API release of Llama Stack, which enables developers to build RAG applications and Agents using tools and safety shields, monitor and those agents with telemetry, and evaluate the agent with scoring functions.

## Context
GenAI application developers need more than just an LLM - they need to integrate tools, connect with their data sources, establish guardrails, and ground the LLM responses effectively. Currently, developers must piece together various tools and APIs, complicating the development lifecycle and increasing costs. The result is that developers are spending more time on these integrations rather than focusing on the application logic itself. The bespoke coupling of components also makes it challenging to adopt state-of-the-art solutions in the rapidly evolving GenAI space. This is particularly difficult for open models like Llama, as best practices are not widely established in the open.

Llama Stack was created to provide developers with a comprehensive and coherent interface that simplifies AI application development and codifies best practices across the Llama ecosystem. Since our launch in September 2024, we have seen a huge uptick in interest in Llama Stack APIs by both AI developers and from partners building AI services with Llama models. Partners like Nvidia, Fireworks, and Ollama have collaborated with us to develop implementations across various APIs, including inference, memory, and safety.

With Llama Stack, you can easily build a RAG agent which can also search the web, do complex math, and custom tool calling. You can use telemetry to inspect those traces, and convert telemetry into evals datasets. And with Llama Stack’s plugin architecture and prepackage distributions, you choose to run your agent anywhere - in the cloud with our partners, deploy your own environment using virtualenv, conda, or Docker, operate locally with Ollama, or even run on mobile devices with our SDKs. Llama Stack offers unprecedented flexibility while also simplifying the developer experience.

## Release
After iterating on the APIs for the last 3 months, today we’re launching a stable release (V1) of the Llama Stack APIs and the corresponding llama-stack server and client packages(v0.1.0). We now have automated tests for providers. These tests make sure that all provider implementations are verified. Developers can now easily and reliably select distributions or providers based on their specific requirements.

There are example standalone apps in llama-stack-apps.


## Key Features of this release

- **Unified API Layer**
  - Inference: Run LLM models
  - RAG: Store and retrieve knowledge for RAG
  - Agents: Build multi-step agentic workflows
  - Tools: Register tools that can be called by the agent
  - Safety: Apply content filtering and safety policies
  - Evaluation: Test model and agent quality
  - Telemetry: Collect and analyze usage data and complex agentic traces
  - Post Training ( Coming Soon ): Fine tune models for specific use cases

- **Rich Provider Ecosystem**
  - Local Development: Meta's Reference, Ollama
  - Cloud: Fireworks, Together, Nvidia, AWS Bedrock, Groq, Cerebras
  - On-premises: Nvidia NIM, vLLM, TGI, Dell-TGI
  - On-device: iOS and Android support

- **Built for Production**
  - Pre-packaged distributions for common deployment scenarios
  - Backwards compatibility across model versions
  - Comprehensive evaluation capabilities
  - Full observability and monitoring

- **Multiple developer interfaces**
  - CLI: Command line interface
  - Python SDK
  - Swift iOS SDK
  - Kotlin Android SDK

- **Sample llama stack applications**
  - Python
  - iOS
  - Android


## New Contributors
* @cdgamarose-nv made their first contribution in https://github.com/meta-llama/llama-stack/pull/661
* @eltociear made their first contribution in https://github.com/meta-llama/llama-stack/pull/675
* @derekslager made their first contribution in https://github.com/meta-llama/llama-stack/pull/692
* @VladOS95-cyber made their first contribution in https://github.com/meta-llama/llama-stack/pull/557
* @frreiss made their first contribution in https://github.com/meta-llama/llama-stack/pull/662
* @pmccarthy made their first contribution in https://github.com/meta-llama/llama-stack/pull/807
* @pandyamarut made their first contribution in https://github.com/meta-llama/llama-stack/pull/362
* @snova-edwardm made their first contribution in https://github.com/meta-llama/llama-stack/pull/555
* @ehhuang made their first contribution in https://github.com/meta-llama/llama-stack/pull/867

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.63...v0.1.0

---

# v0.1.0rc12
Published on: 2025-01-22T22:24:01Z

## New Contributors
* @cdgamarose-nv made their first contribution in https://github.com/meta-llama/llama-stack/pull/661
* @eltociear made their first contribution in https://github.com/meta-llama/llama-stack/pull/675
* @derekslager made their first contribution in https://github.com/meta-llama/llama-stack/pull/692
* @VladOS95-cyber made their first contribution in https://github.com/meta-llama/llama-stack/pull/557
* @frreiss made their first contribution in https://github.com/meta-llama/llama-stack/pull/662
* @pmccarthy made their first contribution in https://github.com/meta-llama/llama-stack/pull/807

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.63...v0.1.0rc11

---

# v0.0.63
Published on: 2024-12-18T07:17:43Z

A small but important bug-fix release to update the URL datatype for the client-SDKs. The issue affected multimodal agentic turns especially.

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.62...v0.0.63

---

# v0.0.62
Published on: 2024-12-18T02:39:43Z

## New Contributors
* @SLR722 made their first contribution in https://github.com/meta-llama/llama-stack/pull/540
* @iamarunbrahma made their first contribution in https://github.com/meta-llama/llama-stack/pull/636

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.61...v0.0.62

---

# v0.0.61
Published on: 2024-12-10T20:50:33Z

## New Contributors
* @sablair made their first contribution in https://github.com/meta-llama/llama-stack/pull/549
* @JeffreyLind3 made their first contribution in https://github.com/meta-llama/llama-stack/pull/547
* @aidando73 made their first contribution in https://github.com/meta-llama/llama-stack/pull/554
* @henrytwo made their first contribution in https://github.com/meta-llama/llama-stack/pull/265
* @sixianyi0721 made their first contribution in https://github.com/meta-llama/llama-stack/pull/507
* @ConnorHack made their first contribution in https://github.com/meta-llama/llama-stack/pull/523
* @yurishkuro made their first contribution in https://github.com/meta-llama/llama-stack/pull/580

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.55...v0.0.61

---

# v0.0.55
Published on: 2024-11-23T17:14:07Z



---

# v0.0.54
Published on: 2024-11-22T00:36:09Z

## New Contributors
* @liyunlu0618 made their first contribution in https://github.com/meta-llama/llama-stack/pull/500

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.53...v0.0.54

---

# v0.0.53
Published on: 2024-11-20T22:18:00Z

🚀  Initial Release Notes for Llama Stack!

### Added
- Resource-oriented design for models, shields, memory banks, datasets and eval tasks
- Persistence for registered objects with distribution
- Ability to persist memory banks created for FAISS
- PostgreSQL KVStore implementation
- Environment variable placeholder support in run.yaml files
- Comprehensive Zero-to-Hero notebooks and quickstart guides
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

## New Contributors
* @Wauplin made their first contribution in https://github.com/meta-llama/llama-stack/pull/9
* @jianyuh made their first contribution in https://github.com/meta-llama/llama-stack/pull/12
* @dltn made their first contribution in https://github.com/meta-llama/llama-stack/pull/14
* @hardikjshah made their first contribution in https://github.com/meta-llama/llama-stack/pull/20
* @raghotham made their first contribution in https://github.com/meta-llama/llama-stack/pull/8
* @jeffxtang made their first contribution in https://github.com/meta-llama/llama-stack/pull/34
* @sisminnmaw made their first contribution in https://github.com/meta-llama/llama-stack/pull/35
* @varunfb made their first contribution in https://github.com/meta-llama/llama-stack/pull/36
* @benjibc made their first contribution in https://github.com/meta-llama/llama-stack/pull/39
* @Nutlope made their first contribution in https://github.com/meta-llama/llama-stack/pull/43
* @hanouticelina made their first contribution in https://github.com/meta-llama/llama-stack/pull/53
* @rsgrewal-aws made their first contribution in https://github.com/meta-llama/llama-stack/pull/96
* @poegej made their first contribution in https://github.com/meta-llama/llama-stack/pull/94
* @abhishekmishragithub made their first contribution in https://github.com/meta-llama/llama-stack/pull/103
* @machina-source made their first contribution in https://github.com/meta-llama/llama-stack/pull/104
* @dijonkitchen made their first contribution in https://github.com/meta-llama/llama-stack/pull/107
* @marklysze made their first contribution in https://github.com/meta-llama/llama-stack/pull/113
* @KarthiDreamr made their first contribution in https://github.com/meta-llama/llama-stack/pull/112
* @delvingdeep made their first contribution in https://github.com/meta-llama/llama-stack/pull/117
* @moldhouse made their first contribution in https://github.com/meta-llama/llama-stack/pull/118
* @bhimrazy made their first contribution in https://github.com/meta-llama/llama-stack/pull/134
* @russellb made their first contribution in https://github.com/meta-llama/llama-stack/pull/128
* @yogishbaliga made their first contribution in https://github.com/meta-llama/llama-stack/pull/105
* @wizardbc made their first contribution in https://github.com/meta-llama/llama-stack/pull/153
* @moritalous made their first contribution in https://github.com/meta-llama/llama-stack/pull/151
* @codefromthecrypt made their first contribution in https://github.com/meta-llama/llama-stack/pull/165
* @AshleyT3 made their first contribution in https://github.com/meta-llama/llama-stack/pull/182
* @Minutis made their first contribution in https://github.com/meta-llama/llama-stack/pull/192
* @prithu-dasgupta made their first contribution in https://github.com/meta-llama/llama-stack/pull/83
* @zainhas made their first contribution in https://github.com/meta-llama/llama-stack/pull/95
* @terrytangyuan made their first contribution in https://github.com/meta-llama/llama-stack/pull/216
* @kebbbnnn made their first contribution in https://github.com/meta-llama/llama-stack/pull/224
* @frntn made their first contribution in https://github.com/meta-llama/llama-stack/pull/247
* @MeDott29 made their first contribution in https://github.com/meta-llama/llama-stack/pull/260
* @tamdogood made their first contribution in https://github.com/meta-llama/llama-stack/pull/261
* @nehal-a2z made their first contribution in https://github.com/meta-llama/llama-stack/pull/275
* @dineshyv made their first contribution in https://github.com/meta-llama/llama-stack/pull/280
* @subramen made their first contribution in https://github.com/meta-llama/llama-stack/pull/286
* @Anush008 made their first contribution in https://github.com/meta-llama/llama-stack/pull/221
* @cheesecake100201 made their first contribution in https://github.com/meta-llama/llama-stack/pull/267
* @heyjustinai made their first contribution in https://github.com/meta-llama/llama-stack/pull/307
* @sacmehta made their first contribution in https://github.com/meta-llama/llama-stack/pull/326
* @stevegrubb made their first contribution in https://github.com/meta-llama/llama-stack/pull/349
* @hickeyma made their first contribution in https://github.com/meta-llama/llama-stack/pull/456
* @vladimirivic made their first contribution in https://github.com/meta-llama/llama-stack/pull/465
* @wukaixingxp made their first contribution in https://github.com/meta-llama/llama-stack/pull/471
* @Riandy made their first contribution in https://github.com/meta-llama/llama-stack/pull/476
* @mattf made their first contribution in https://github.com/meta-llama/llama-stack/pull/470
* @chuenlok made their first contribution in https://github.com/meta-llama/llama-stack/pull/467
* @iseeyuan made their first contribution in https://github.com/meta-llama/llama-stack/pull/485

**Full Changelog**: https://github.com/meta-llama/llama-stack/commits/v0.0.53

---
