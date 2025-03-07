# Changelog

# v0.1.5.1
Published on: 2025-02-28T22:37:44Z

## What's Changed
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

## All changes
* test: add a ci-tests distro template for running e2e tests by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1237
* refactor: combine start scripts for each env by @cdoern in https://github.com/meta-llama/llama-stack/pull/1139
* fix: pre-commit updates by @cdoern in https://github.com/meta-llama/llama-stack/pull/1243
* fix: Update getting_started.ipynb by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1245
* fix: Update Llama_Stack_Benchmark_Evals.ipynb by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1246
* build: hint on Python version for uv venv by @leseb in https://github.com/meta-llama/llama-stack/pull/1172
* fix: include timezone in Agent steps' timestamps by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1247
* LocalInferenceImpl update for LS013 by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/1242
* fix: Raise exception when tool call result is None by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1253
* fix: resolve type hint issues and import dependencies by @leseb in https://github.com/meta-llama/llama-stack/pull/1176
* fix: build_venv expects an extra argument by @cdoern in https://github.com/meta-llama/llama-stack/pull/1233
* feat: completing text /chat-completion and /completion tests by @LESSuseLESS in https://github.com/meta-llama/llama-stack/pull/1223
* fix: update index.md to include 0.1.4 by @raghotham in https://github.com/meta-llama/llama-stack/pull/1259
* docs: Remove $ from client CLI ref  to add valid copy and paste ability by @kelbrown20 in https://github.com/meta-llama/llama-stack/pull/1260
* feat: Add Groq distribution template by @VladOS95-cyber in https://github.com/meta-llama/llama-stack/pull/1173
* chore: update the zero_to_hero_guide doc link by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1220
* build: Merge redundant "files" field for codegen check in .pre-commit-config.yaml by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1261
* refactor(server): replace print statements with logger by @leseb in https://github.com/meta-llama/llama-stack/pull/1250
* fix: fix the describe table display issue by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1221
* chore: update download error message by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1217
* chore: removed executorch submodule by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/1265
* refactor: move OpenAI compat utilities from nvidia to openai_compat by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1258
* feat: add (openai, anthropic, gemini) providers via litellm by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1267
* feat: [post training] support save hf safetensor format checkpoint by @SLR722 in https://github.com/meta-llama/llama-stack/pull/845
* fix: the pre-commit new line issue by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1272
* fix(cli): Missing default for --image-type in stack run command by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1274
* fix: Get builtin tool calling working in remote-vllm by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1236
* feat: remove special handling of builtin::rag tool by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1015
* feat: update the post training notebook by @SLR722 in https://github.com/meta-llama/llama-stack/pull/1280
* fix: time logging format by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1281
* feat: allow specifying specific tool within toolgroup by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1239
* fix: sqlite conn by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1282
* chore: upgrade uv pre-commit version, uv-sync -> uv-lock by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1284
* fix: don't attempt to clean gpu memory up when device is cpu by @booxter in https://github.com/meta-llama/llama-stack/pull/1191
* feat: Add model context protocol tools with ollama provider by @Shreyanand in https://github.com/meta-llama/llama-stack/pull/1283
* fix(test): update client-sdk tests to handle tool format parametrization better by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1287
* feat: add nemo retriever text embedding models to nvidia inference provider by @mattf in https://github.com/meta-llama/llama-stack/pull/1218
* feat: don't silently ignore incorrect toolgroup by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1285
* feat: ability to retrieve agents session, turn, step by ids by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1286
* fix(test): no need to specify tool prompt format explicitly in tests by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1295
* chore: remove vector_db_id from AgentSessionInfo by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1296
* fix: Revert "chore: remove vector_db_id from AgentSessionInfo" by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1299
* feat(providers): Groq now uses LiteLLM openai-compat by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1303
* fix: duplicate ToolResponseMessage in Turn message history by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1305
* fix: don't include tool args not in the function definition by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1307
* fix: update notebooks to avoid using the nutsy --image-name __system__ thing by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1308
* fix: register provider model name and HF alias in run.yaml by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1304
* build: Add dotenv file for running tests with uv by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1251
* docs: update the output of llama-stack-client models list by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1271
* fix: Avoid unexpected keyword argument for sentence_transformers by @luis5tb in https://github.com/meta-llama/llama-stack/pull/1269
* feat: add nvidia embedding implementation for new signature, task_type, output_dimention, text_truncation by @mattf in https://github.com/meta-llama/llama-stack/pull/1213
* chore: add subcommands description in help by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1219
* fix: Structured outputs for recursive models by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1311
* fix: litellm tool call parsing event type to in_progress by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1312
* fix: Incorrect import path for print_subcommand_description() by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1313
* fix: Incorrect import path for print_subcommand_description() by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1314
* fix: Incorrect import path for print_subcommand_description() by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1315
* test: Only run embedding tests for remote::nvidia by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1317
* fix: update getting_started notebook to pass nbeval by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1318
* fix: [Litellm]Do not swallow first token by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1316
* feat: update the default system prompt for 3.2/3.3 models by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1310
* fix: Agent telemetry inputs/outputs should be structured by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1302
* fix: check conda env name using basepath in exec.py by @dineshyv in https://github.com/meta-llama/llama-stack/pull/1301

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


## What's Changed
* build: resync uv and deps on 0.1.3 by @leseb in https://github.com/meta-llama/llama-stack/pull/1108
* style: fix the capitalization issue by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1117
* feat: log start, complete time to Agent steps by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1116
* fix: Ensure a tool call can be converted before adding to buffer by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1119
* docs: Fix incorrect link and command for generating API reference by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1124
* chore: remove --no-list-templates option by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1121
* style: update verify-download help text by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1134
* style: update download help text by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1135
* fix: modify the model id title for model list by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1095
* fix: direct client pydantic type casting by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1145
* style: remove prints in codebase by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1146
* feat: support tool_choice = {required, none, <function>} by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1059
* test: Enable test_text_chat_completion_with_tool_choice_required for remote::vllm by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1148
* fix(rag-example): add provider_id to avoid llama_stack_client 400 error by @fulvius31 in https://github.com/meta-llama/llama-stack/pull/1114
* fix: Get distro_codegen.py working with default deps and enabled in pre-commit hooks by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1123
* chore: remove llama_models.llama3.api imports from providers by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1107
* docs: fix Python llama_stack_client SDK links by @leseb in https://github.com/meta-llama/llama-stack/pull/1150
* feat: Chunk sqlite-vec writes by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/1094
* fix: miscellaneous job management improvements in torchtune by @booxter in https://github.com/meta-llama/llama-stack/pull/1136
* feat: add aggregation_functions to llm_as_judge_405b_simpleqa by @SLR722 in https://github.com/meta-llama/llama-stack/pull/1164
* feat: inference passthrough provider  by @SLR722 in https://github.com/meta-llama/llama-stack/pull/1166
* docs: Remove unused python-openapi and json-strong-typing in openapi_generator by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1167
* docs: improve API contribution guidelines by @leseb in https://github.com/meta-llama/llama-stack/pull/1137
* feat: add a option to list the downloaded models by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1127
* fix: Fixing some small issues with the build scripts by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/1132
* fix: llama stack build use UV_SYSTEM_PYTHON to install dependencies to system environment by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1163
* build: add missing dev dependencies for unit tests by @leseb in https://github.com/meta-llama/llama-stack/pull/1004
* fix: More robust handling of the arguments in tool call response in remote::vllm by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1169
* Added support for mongoDB KV store by @shrinitg in https://github.com/meta-llama/llama-stack/pull/543
* script for running client sdk tests by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/895
* test: skip model registration for unsupported providers by @leseb in https://github.com/meta-llama/llama-stack/pull/1030
* feat: Enable CPU training for torchtune by @booxter in https://github.com/meta-llama/llama-stack/pull/1140
* fix: add logging import by @raspawar in https://github.com/meta-llama/llama-stack/pull/1174
* docs: Add note about distro_codegen.py and provider dependencies by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1175
* chore: slight renaming of model alias stuff by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1181
* feat: adding endpoints for files and uploads by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/1070
* docs: Fix Links, Add Podman Instructions, Vector DB Unregister, and Example Script by @kevincogan in https://github.com/meta-llama/llama-stack/pull/1129
* chore!: deprecate eval/tasks by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1186
* fix: some telemetry APIs don't currently work by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1188
* feat: D69478008 [llama-stack] turning tests into data-driven by @LESSuseLESS in https://github.com/meta-llama/llama-stack/pull/1180
* feat: register embedding models for ollama, together, fireworks by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1190
* feat(providers): add NVIDIA Inference embedding provider and tests by @mattf in https://github.com/meta-llama/llama-stack/pull/935
* docs: Add missing uv command for docs generation in contributing guide by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1197
* docs: Simplify installation guide with `uv` by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1196
* fix: BuiltinTool JSON serialization in remote vLLM provider by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1183
* ci: improve GitHub Actions workflow for website builds by @leseb in https://github.com/meta-llama/llama-stack/pull/1151
* fix: pass tool_prompt_format to chat_formatter by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1198
* fix(api): update embeddings signature so inputs and outputs list align by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1161
* feat(api): Add options for supporting various embedding models by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1192
* fix: update URL import, URL -> ImageContentItemImageURL by @mattf in https://github.com/meta-llama/llama-stack/pull/1204
* feat: model remove cmd by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1128
* chore: remove configure subcommand by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1202
* fix: remove list of list tests, no longer relevant after #1161 by @mattf in https://github.com/meta-llama/llama-stack/pull/1205
* test(client-sdk): Update embedding test types to use latest imports by @raspawar in https://github.com/meta-llama/llama-stack/pull/1203
* fix: convert back to model descriptor for model in list --downloaded by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1201
* docs: Add missing uv command and clarify website rebuild by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1199
* fix: Updating images so that they are able to run without root access by @jland-redhat in https://github.com/meta-llama/llama-stack/pull/1208
* fix: pull ollama embedding model if necessary by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1209
* chore: move embedding deps to RAG tool where they are needed by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1210
* feat(1/n): api: unify agents for handling server & client tools by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1178
* feat: tool outputs metadata by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1155
* ci: add mypy for static type checking by @leseb in https://github.com/meta-llama/llama-stack/pull/1101
* feat(providers): support non-llama models for inference providers by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1200
* test: fix test_rag_agent test by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1215
* feat: add substring search for model list by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1099
* test: do not overwrite agent_config by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1216
* docs: Adding Provider sections to docs by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/1195
* fix: update virtualenv building so llamastack- prefix is not added, make notebook experience easier by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1225
* feat: add --run to llama stack build by @cdoern in https://github.com/meta-llama/llama-stack/pull/1156
* docs: Add vLLM to the list of inference providers in concepts and providers pages by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1227
* docs: small fixes by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1224
* fix: avoid failure when no special pip deps and better exit by @leseb in https://github.com/meta-llama/llama-stack/pull/1228
* fix: set default tool_prompt_format in inference api by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1214
* test: fix test_tool_choice by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1234

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

## What's Changed
* Getting started notebook update by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/936
* docs: update index.md for 0.1.2 by @raghotham in https://github.com/meta-llama/llama-stack/pull/1013
* test: Make text-based chat completion tests run 10x faster by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1016
* chore: Updated requirements.txt by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/1017
* test: Use JSON tool prompt format for remote::vllm provider by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1019
* docs: Render check marks correctly on PyPI by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1024
* docs: update rag.md example code to prevent errors by @MichaelClifford in https://github.com/meta-llama/llama-stack/pull/1009
* build: update uv lock to sync package versions by @leseb in https://github.com/meta-llama/llama-stack/pull/1026
* fix: Gaps in doc codegen by @ellistarn in https://github.com/meta-llama/llama-stack/pull/1035
* fix: Readthedocs cannot parse comments, resulting in docs bugs by @ellistarn in https://github.com/meta-llama/llama-stack/pull/1033
* fix: a bad newline in ollama docs by @ellistarn in https://github.com/meta-llama/llama-stack/pull/1036
* fix: Update Qdrant support post-refactor by @jwm4 in https://github.com/meta-llama/llama-stack/pull/1022
* test: replace blocked image URLs with GitHub-hosted by @leseb in https://github.com/meta-llama/llama-stack/pull/1025
* fix: Added missing `tool_config` arg in SambaNova `chat_completion()` by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1042
* docs: Updating wording and nits in the README.md by @kelbrown20 in https://github.com/meta-llama/llama-stack/pull/992
* docs: remove changelog mention from PR template by @leseb in https://github.com/meta-llama/llama-stack/pull/1049
* docs: reflect actual number of spaces for indent by @booxter in https://github.com/meta-llama/llama-stack/pull/1052
* fix: agent config validation by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1053
* feat: add MetricResponseMixin to chat completion response types by @dineshyv in https://github.com/meta-llama/llama-stack/pull/1050
* feat: make telemetry attributes be dict[str,PrimitiveType] by @dineshyv in https://github.com/meta-llama/llama-stack/pull/1055
* fix: filter out remote::sample providers when listing by @booxter in https://github.com/meta-llama/llama-stack/pull/1057
* feat: Support tool calling for non-streaming chat completion in remote vLLM provider by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1034
* perf: ensure ToolCall in ChatCompletionResponse is subset of ChatCompletionRequest.tools by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1041
* chore: update return type to Optional[str] by @leseb in https://github.com/meta-llama/llama-stack/pull/982
* feat: Support tool calling for streaming chat completion in remote vLLM provider by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1063
* fix: show proper help text by @cdoern in https://github.com/meta-llama/llama-stack/pull/1065
* feat: add support for running in a venv by @cdoern in https://github.com/meta-llama/llama-stack/pull/1018
* feat: Adding sqlite-vec as a vectordb by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/1040
* feat: support listing all for `llama stack list-providers` by @booxter in https://github.com/meta-llama/llama-stack/pull/1056
* docs: Mention convential commits format in CONTRIBUTING.md by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1075
* fix: logprobs support in remote-vllm provider by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1074
* fix: improve signal handling and update dependencies by @leseb in https://github.com/meta-llama/llama-stack/pull/1044
* style: update model id in model list title by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1072
* fix: make backslash work in GET /models/{model_id:path} by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1068
* chore: Link to Groq docs in the warning message for preview model by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1060
* fix: remove :path in agents by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1077
* build: format codebase imports using ruff linter by @leseb in https://github.com/meta-llama/llama-stack/pull/1028
* chore: Consistent naming for VectorIO providers by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1023
* test: Enable logprobs top_k tests for remote::vllm by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1080
* docs: Fix url to the llama-stack-spec yaml/html files by @vishnoianil in https://github.com/meta-llama/llama-stack/pull/1081
* fix: Update VectorIO config classes in registry by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1079
* test: Add qdrant to provider tests by @jwm4 in https://github.com/meta-llama/llama-stack/pull/1039
* test: add test for Agent.create_turn non-streaming response by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1078
* fix!: update eval-tasks -> benchmarks by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1032
* fix: openapi for eval-task by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1085
* fix: regex pattern matching to support :path suffix in the routes by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/1089
* fix: disable sqlite-vec test by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/1090
* fix: add the missed help description info by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1096
* fix: Update QdrantConfig to QdrantVectorIOConfig by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1104
* docs: Add region parameter to Bedrock provider by @raghotham in https://github.com/meta-llama/llama-stack/pull/1103
* build: configure ruff from pyproject.toml by @leseb in https://github.com/meta-llama/llama-stack/pull/1100
* chore: move all Llama Stack types from llama-models to llama-stack by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1098
* fix: enable_session_persistence in AgentConfig should be optional by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1012
* fix: improve stack build on venv by @leseb in https://github.com/meta-llama/llama-stack/pull/980
* fix: remove the empty line by @reidliu41 in https://github.com/meta-llama/llama-stack/pull/1097

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

## What's Changed
* Fix UBI9 image build when installing Python packages via uv by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/926
* Fix precommit check after moving to ruff by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/927
* LocalInferenceImpl update for LS 0.1 by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/911
* Properly close PGVector DB connection during shutdown() by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/931
* Add issue template config with docs and Discord links by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/930
* Fix uv pip install timeout issue for PyTorch by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/929
* github: ignore non-hidden python virtual environments by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/939
* fix: broken link in Quick Start doc by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/943
* fix: broken "core concepts" link in docs website by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/940
* Misc fixes by @ashwinb in https://github.com/meta-llama/llama-stack/pull/944
* fix: formatting for ollama note in Quick Start doc by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/945
* [docs] typescript sdk readme by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/946
* Support sys_prompt behavior in inference by @ehhuang in https://github.com/meta-llama/llama-stack/pull/937
* if client.initialize fails, the example should exit by @cdoern in https://github.com/meta-llama/llama-stack/pull/954
* Add Podman instructions to Quick Start by @jwm4 in https://github.com/meta-llama/llama-stack/pull/957
* github: issue templates automatically apply relevant label by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/956
* docs: miscellaneous small fixes by @booxter in https://github.com/meta-llama/llama-stack/pull/961
* Make a couple properties optional by @ashwinb in https://github.com/meta-llama/llama-stack/pull/963
* [docs] Make RAG example self-contained by @booxter in https://github.com/meta-llama/llama-stack/pull/962
* docs, tests: replace datasets.rst with memory_optimizations.rst by @booxter in https://github.com/meta-llama/llama-stack/pull/968
* Fix broken pgvector provider and memory leaks by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/947
* [docs] update the zero_to_hero_guide llama stack version to 0.1.0 by @kami619 in https://github.com/meta-llama/llama-stack/pull/960
* missing T in import by @cooktheryan in https://github.com/meta-llama/llama-stack/pull/974
* Fix README.md notebook links by @aakankshaduggal in https://github.com/meta-llama/llama-stack/pull/976
* docs: clarify host.docker.internal works for recent podman by @booxter in https://github.com/meta-llama/llama-stack/pull/977
* docs: add addn server guidance for Linux users in Quick Start by @nathan-weinberg in https://github.com/meta-llama/llama-stack/pull/972
* sys_prompt support in Agent by @ehhuang in https://github.com/meta-llama/llama-stack/pull/938
* chore: update PR template to reinforce changelog by @leseb in https://github.com/meta-llama/llama-stack/pull/988
* github: update PR template to use correct syntax to auto-close issues by @booxter in https://github.com/meta-llama/llama-stack/pull/989
* chore: remove unused argument by @cdoern in https://github.com/meta-llama/llama-stack/pull/987
* test: replace memory with vector_io fixture by @leseb in https://github.com/meta-llama/llama-stack/pull/984
* docs: use uv in CONTRIBUTING guide by @leseb in https://github.com/meta-llama/llama-stack/pull/970
* docs: Add license badge to README.md by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/994
* Add Kubernetes deployment guide by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/899
* Fix incorrect handling of chat completion endpoint in remote::vLLM by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/951
* ci: Add semantic PR title check by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/979
* feat: Add a new template for `dell` by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/978
* docs: Correct typos in Zero to Hero guide by @mlecanu in https://github.com/meta-llama/llama-stack/pull/997
* fix: Update rag examples to use fresh faiss index every time by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/998
* doc: getting started notebook by @ehhuang in https://github.com/meta-llama/llama-stack/pull/996
* test: fix flaky agent test by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1002
* test: rm unused exception alias in pytest.raises by @leseb in https://github.com/meta-llama/llama-stack/pull/991
* fix: List providers command prints out non-existing APIs from registry. Fixes #966 by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/969
* chore: add missing ToolConfig import in groq.py by @leseb in https://github.com/meta-llama/llama-stack/pull/983
* test: remove flaky agent test by @ehhuang in https://github.com/meta-llama/llama-stack/pull/1006
* test: Split inference tests to text and vision by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/1008
* feat: Add HTTPS serving option by @ashwinb in https://github.com/meta-llama/llama-stack/pull/1000
* test: encode image data as base64 by @leseb in https://github.com/meta-llama/llama-stack/pull/1003
* fix: Ensure a better error stack trace when llama-stack is not built by @cdoern in https://github.com/meta-llama/llama-stack/pull/950
* refactor(ollama): model availability check by @leseb in https://github.com/meta-llama/llama-stack/pull/986

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

## What's Changed
* Update doc templates for running safety on self-hosted templates by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/874
* Update GH action so it correctly queries for test.pypi, etc. by @ashwinb in https://github.com/meta-llama/llama-stack/pull/875
* Fix report generation for url endpoints by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/876
* Fixed typo by @BakungaBronson in https://github.com/meta-llama/llama-stack/pull/877
* Fixed multiple typos by @BakungaBronson in https://github.com/meta-llama/llama-stack/pull/878
* Ensure llama stack build --config <> --image-type <> works by @ashwinb in https://github.com/meta-llama/llama-stack/pull/879
* Update documentation by @ashwinb in https://github.com/meta-llama/llama-stack/pull/865
* Update discriminator to have the correct `mapping` by @ashwinb in https://github.com/meta-llama/llama-stack/pull/881
* Fix telemetry init by @dineshyv in https://github.com/meta-llama/llama-stack/pull/885
* Sambanova - LlamaGuard by @snova-edwardm in https://github.com/meta-llama/llama-stack/pull/886
* Update index.md by @Ckhanoyan in https://github.com/meta-llama/llama-stack/pull/888
* Report generation minor fixes by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/884
* adding readme to docs folder for easier discoverability of notebooks … by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/857
* Agent response format by @hanzlfs in https://github.com/meta-llama/llama-stack/pull/660
* Add windows support for build execution by @VladOS95-cyber in https://github.com/meta-llama/llama-stack/pull/889
* Add run win command for stack by @VladOS95-cyber in https://github.com/meta-llama/llama-stack/pull/890
* Use ruamel.yaml to format the OpenAPI spec by @ashwinb in https://github.com/meta-llama/llama-stack/pull/892
* Fix Chroma adapter by @ashwinb in https://github.com/meta-llama/llama-stack/pull/893
* align with CompletionResponseStreamChunk.delta as str (instead of TextDelta) by @mattf in https://github.com/meta-llama/llama-stack/pull/900
* add NVIDIA_BASE_URL and NVIDIA_API_KEY to control hosted vs local endpoints by @mattf in https://github.com/meta-llama/llama-stack/pull/897
* Fix validator of "container" image type by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/901
* Update OpenAPI generator to add param and field documentation by @ashwinb in https://github.com/meta-llama/llama-stack/pull/896
* Fix link to selection guide and change "docker" to "container" by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/898
* [#432] Groq Provider tool call tweaks by @aidando73 in https://github.com/meta-llama/llama-stack/pull/811
* Fix running stack built with base conda environment by @dvrogozh in https://github.com/meta-llama/llama-stack/pull/903
* create a github action for triggering client-sdk tests on new pull-request by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/850
* log probs - mark pytests as xfail for unsupported providers + add support for together by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/883
* SambaNova supports Llama 3.3 by @snova-edwardm in https://github.com/meta-llama/llama-stack/pull/905
* fix ImageContentItem to take base64 string as image.data by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/909
* Fix Agents to support code and rag simultaneously by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/908
* add test for user message w/ image.data content by @mattf in https://github.com/meta-llama/llama-stack/pull/906
* openapi gen return type fix for streaming/non-streaming by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/910
* feat: enable xpu support for meta-reference stack by @dvrogozh in https://github.com/meta-llama/llama-stack/pull/558
* Sec fixes as raised by bandit by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/917
* Run code-gen by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/916
* fix rag tests by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/918
* Use `uv pip install` instead of `pip install` by @ashwinb in https://github.com/meta-llama/llama-stack/pull/921
* add image support to NVIDIA inference provider by @mattf in https://github.com/meta-llama/llama-stack/pull/907

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


### What's Changed
* [4/n][torchtune integration] support lazy load model during inference by @SLR722 in https://github.com/meta-llama/llama-stack/pull/620
* remove unused telemetry related code for console by @dineshyv in https://github.com/meta-llama/llama-stack/pull/659
* Fix Meta reference GPU implementation by @ashwinb in https://github.com/meta-llama/llama-stack/pull/663
* Fixed imports for inference by @cdgamarose-nv in https://github.com/meta-llama/llama-stack/pull/661
* fix trace starting in library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/655
* Add Llama 70B 3.3 to fireworks by @aidando73 in https://github.com/meta-llama/llama-stack/pull/654
* Tools API with brave and MCP providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/639
* [torchtune integration] post training + eval by @SLR722 in https://github.com/meta-llama/llama-stack/pull/670
* Fix post training apis broken by torchtune release by @SLR722 in https://github.com/meta-llama/llama-stack/pull/674
* Add missing venv option in --image-type by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/677
* Removed unnecessary CONDA_PREFIX env var in installation guide by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/683
* Add 3.3 70B to Ollama inference provider by @aidando73 in https://github.com/meta-llama/llama-stack/pull/681
* docs: update evals_reference/index.md by @eltociear in https://github.com/meta-llama/llama-stack/pull/675
* [remove import *][1/n] clean up import & in apis/* by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/689
* [bugfix] fix broken vision inference, change serialization for bytes by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/693
* Minor Quick Start documentation updates. by @derekslager in https://github.com/meta-llama/llama-stack/pull/692
* [bugfix] fix meta-reference agents w/ safety multiple model loading pytest by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/694
* [bugfix] fix prompt_adapter interleaved_content_convert_to_raw  by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/696
* Add missing "inline::" prefix for providers in building_distro.md by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/702
* Fix failing flake8 E226 check by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/701
* Add missing newlines before printing the Dockerfile content by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/700
* Add JSON structured outputs to Ollama Provider by @aidando73 in https://github.com/meta-llama/llama-stack/pull/680
* [#407] Agents: Avoid calling tools that haven't been explicitly enabled by @aidando73 in https://github.com/meta-llama/llama-stack/pull/637
* Made changes to readme and pinning to llamastack v0.0.61 by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/624
* [rag evals][1/n] refactor base scoring fn & data schema check by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/664
* [Post Training] Fix missing import by @SLR722 in https://github.com/meta-llama/llama-stack/pull/705
* Import from the right path  by @SLR722 in https://github.com/meta-llama/llama-stack/pull/708
* [#432] Add Groq Provider - chat completions by @aidando73 in https://github.com/meta-llama/llama-stack/pull/609
* Change post training run.yaml inference config  by @SLR722 in https://github.com/meta-llama/llama-stack/pull/710
* [Post training] make validation steps configurable by @SLR722 in https://github.com/meta-llama/llama-stack/pull/715
* Fix incorrect entrypoint for broken `llama stack run` by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/706
* Fix assert message and call to completion_request_to_prompt in remote:vllm by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/709
* Fix Groq invalid self.config reference by @aidando73 in https://github.com/meta-llama/llama-stack/pull/719
* support llama3.1 8B instruct in post training by @SLR722 in https://github.com/meta-llama/llama-stack/pull/698
* remove default logger handlers when using libcli with notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/718
* move DataSchemaValidatorMixin into standalone utils by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/720
* add 3.3 to together inference provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/729
* Update CODEOWNERS - add sixianyi0721 as the owner by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/731
* fix links for distro by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/733
* add --version to llama stack CLI & /version endpoint by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/732
* agents to use tools api by @dineshyv in https://github.com/meta-llama/llama-stack/pull/673
* Add X-LlamaStack-Client-Version, rename ProviderData -> Provider-Data by @ashwinb in https://github.com/meta-llama/llama-stack/pull/735
* Check version incompatibility by @ashwinb in https://github.com/meta-llama/llama-stack/pull/738
* Add persistence for localfs datasets by @VladOS95-cyber in https://github.com/meta-llama/llama-stack/pull/557
* Fixed typo in default VLLM_URL in remote-vllm.md by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/723
* Consolidating Memory tests under client-sdk by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/703
* Expose LLAMASTACK_PORT in cli.stack.run by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/722
* remove conflicting default for tool prompt format in chat completion by @dineshyv in https://github.com/meta-llama/llama-stack/pull/742
* rename LLAMASTACK_PORT to LLAMA_STACK_PORT for consistency with other env vars by @raghotham in https://github.com/meta-llama/llama-stack/pull/744
* Add inline vLLM inference provider to regression tests and fix regressions by @frreiss in https://github.com/meta-llama/llama-stack/pull/662
* [CICD] github workflow to push nightly package to testpypi by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/734
* Replaced zrangebylex method in the range method by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/521
* Improve model download doc by @SLR722 in https://github.com/meta-llama/llama-stack/pull/748
* Support building UBI9 base container image by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/676
* update notebook to use new tool defs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/745
* Add provider data passing for library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/750
* [Fireworks] Update model name for Fireworks by @benjibc in https://github.com/meta-llama/llama-stack/pull/753
* Consolidating Inference tests under client-sdk tests by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/751
* Consolidating Safety tests from various places under client-sdk by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/699
* [CI/CD] more robust re-try for downloading testpypi package by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/749
* [#432] Add Groq Provider - tool calls by @aidando73 in https://github.com/meta-llama/llama-stack/pull/630
* Rename ipython to tool by @ashwinb in https://github.com/meta-llama/llama-stack/pull/756
* Fix incorrect Python binary path for UBI9 image by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/757
* Update Cerebras docs to include header by @henrytwo in https://github.com/meta-llama/llama-stack/pull/704
* Add init files to post training folders by @SLR722 in https://github.com/meta-llama/llama-stack/pull/711
* Switch to use importlib instead of deprecated pkg_resources by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/678
* [bugfix] fix streaming GeneratorExit exception with LlamaStackAsLibraryClient by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/760
* Fix telemetry to work on reinstantiating new lib cli by @dineshyv in https://github.com/meta-llama/llama-stack/pull/761
* [post training]  define llama stack post training dataset format by @SLR722 in https://github.com/meta-llama/llama-stack/pull/717
* add braintrust to experimental-post-training template by @SLR722 in https://github.com/meta-llama/llama-stack/pull/763
* added support of PYPI_VERSION in stack build by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/762
* Fix broken tests in test_registry by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/707
* Fix fireworks run-with-safety template by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/766
* Free up memory after post training finishes by @SLR722 in https://github.com/meta-llama/llama-stack/pull/770
* Fix issue when generating distros by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/755
* Convert `SamplingParams.strategy` to a union by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/767
* [CICD] Github workflow for publishing Docker images by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/764
* [bugfix] fix llama guard parsing ContentDelta by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/772
* rebase eval test w/ tool_runtime fixtures by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/773
* More idiomatic REST API by @dineshyv in https://github.com/meta-llama/llama-stack/pull/765
* add nvidia distribution by @cdgamarose-nv in https://github.com/meta-llama/llama-stack/pull/565
* bug fixes on inference tests by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/774
* [bugfix] fix inference sdk test for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/775
* fix routing in library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/776
* [bugfix] fix client-sdk tests for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/777
* fix nvidia inference provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/781
* Make notebook testable by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/780
* Fix telemetry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/787
* fireworks add completion logprobs adapter by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/778
* Idiomatic REST API: Inspect by @dineshyv in https://github.com/meta-llama/llama-stack/pull/779
* Idiomatic REST API: Evals by @dineshyv in https://github.com/meta-llama/llama-stack/pull/782
* Add notebook testing to nightly build job by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/785
* [test automation] support run tests on config file  by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/730
* Idiomatic REST API: Telemetry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/786
* Make llama stack build not create a new conda by default by @ashwinb in https://github.com/meta-llama/llama-stack/pull/788
* REST API fixes by @dineshyv in https://github.com/meta-llama/llama-stack/pull/789
* fix cerebras template by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/790
* [Test automation] generate custom test report by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/739
* cerebras template update for memory by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/792
* Pin torchtune pkg version by @SLR722 in https://github.com/meta-llama/llama-stack/pull/791
* fix the code execution test in sdk tests by @dineshyv in https://github.com/meta-llama/llama-stack/pull/794
* add default toolgroups to all providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/795
* Fix tgi adapter by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/796
* Remove llama-guard in Cerebras template & improve agent test by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/798
* meta reference inference fixes by @ashwinb in https://github.com/meta-llama/llama-stack/pull/797
* fix provider model list test by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/800
* fix playground for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/799
* fix eval notebook & add test to workflow by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/803
* add json_schema_type to ParamType deps by @dineshyv in https://github.com/meta-llama/llama-stack/pull/808
* Fixing small typo in quick start guide by @pmccarthy in https://github.com/meta-llama/llama-stack/pull/807
* cannot import name 'GreedySamplingStrategy' by @aidando73 in https://github.com/meta-llama/llama-stack/pull/806
* optional api dependencies by @ashwinb in https://github.com/meta-llama/llama-stack/pull/793
* fix vllm template by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/813
* More generic image type for OCI-compliant container technologies by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/802
* add mcp runtime as default to all providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/816
* fix vllm base64 image inference by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/815
* fix again vllm for non base64 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/818
* Fix incorrect RunConfigSettings due to the removal of conda_env by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/801
* Fix incorrect image type in publish-to-docker workflow by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/819
* test report for v0.1 by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/814
* [CICD] add simple test step for docker build workflow, fix prefix bug by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/821
* add section for mcp tool usage in notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/831
* [ez] structured output for /completion ollama & enable tests by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/822
* add pytest option to generate a functional report for distribution by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/833
* bug fix for distro report generation by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/836
* [memory refactor][1/n] Rename Memory -> VectorIO, MemoryBanks -> VectorDBs by @ashwinb in https://github.com/meta-llama/llama-stack/pull/828
* [memory refactor][2/n] Update faiss and make it pass tests by @ashwinb in https://github.com/meta-llama/llama-stack/pull/830
* [memory refactor][3/n] Introduce RAGToolRuntime as a specialized sub-protocol by @ashwinb in https://github.com/meta-llama/llama-stack/pull/832
* [memory refactor][4/n] Update the client-sdk test for RAG by @ashwinb in https://github.com/meta-llama/llama-stack/pull/834
* [memory refactor][5/n] Migrate all vector_io providers by @ashwinb in https://github.com/meta-llama/llama-stack/pull/835
* [memory refactor][6/n] Update naming and routes by @ashwinb in https://github.com/meta-llama/llama-stack/pull/839
* Fix fireworks client sdk chat completion with images by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/840
* [inference api] modify content types so they follow a more standard structure by @ashwinb in https://github.com/meta-llama/llama-stack/pull/841
* fix experimental-post-training template by @SLR722 in https://github.com/meta-llama/llama-stack/pull/842
* Improved report generation for providers by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/844
* [client sdk test] add options for inference_model, safety_shield, embedding_model by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/843
* add distro report by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/847
* Update Documentation by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/838
* Update OpenAPI generator to output discriminator by @ashwinb in https://github.com/meta-llama/llama-stack/pull/848
* update docs for tools and telemetry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/846
* Add vLLM raw completions API by @aidando73 in https://github.com/meta-llama/llama-stack/pull/823
* update doc for client-sdk testing  by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/849
* Delete docs/to_situate directory by @raghotham in https://github.com/meta-llama/llama-stack/pull/851
* Fixed distro documentation by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/852
* remove getting started notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/853
* More Updates to Read the Docs  by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/856
* Llama_Stack_Building_AI_Applications.ipynb -> getting_started.ipynb by @dineshyv in https://github.com/meta-llama/llama-stack/pull/854
* update docs for adding new API providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/855
* Add Runpod Provider + Distribution by @pandyamarut in https://github.com/meta-llama/llama-stack/pull/362
* Sambanova inference provider by @snova-edwardm in https://github.com/meta-llama/llama-stack/pull/555
* Updates to ReadTheDocs by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/859
* sync readme.md to index.md by @dineshyv in https://github.com/meta-llama/llama-stack/pull/860
* More updates to ReadTheDocs by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/861
* make default tool prompt format none in agent config by @dineshyv in https://github.com/meta-llama/llama-stack/pull/863
* update the client reference by @dineshyv in https://github.com/meta-llama/llama-stack/pull/864
* update python sdk reference by @dineshyv in https://github.com/meta-llama/llama-stack/pull/866
* remove logger handler only in notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/868
* Update 'first RAG agent' in gettingstarted doc by @ehhuang in https://github.com/meta-llama/llama-stack/pull/867

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

## What's Changed
* [4/n][torchtune integration] support lazy load model during inference by @SLR722 in https://github.com/meta-llama/llama-stack/pull/620
* remove unused telemetry related code for console by @dineshyv in https://github.com/meta-llama/llama-stack/pull/659
* Fix Meta reference GPU implementation by @ashwinb in https://github.com/meta-llama/llama-stack/pull/663
* Fixed imports for inference by @cdgamarose-nv in https://github.com/meta-llama/llama-stack/pull/661
* fix trace starting in library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/655
* Add Llama 70B 3.3 to fireworks by @aidando73 in https://github.com/meta-llama/llama-stack/pull/654
* Tools API with brave and MCP providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/639
* [torchtune integration] post training + eval by @SLR722 in https://github.com/meta-llama/llama-stack/pull/670
* Fix post training apis broken by torchtune release by @SLR722 in https://github.com/meta-llama/llama-stack/pull/674
* Add missing venv option in --image-type by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/677
* Removed unnecessary CONDA_PREFIX env var in installation guide by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/683
* Add 3.3 70B to Ollama inference provider by @aidando73 in https://github.com/meta-llama/llama-stack/pull/681
* docs: update evals_reference/index.md by @eltociear in https://github.com/meta-llama/llama-stack/pull/675
* [remove import *][1/n] clean up import & in apis/* by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/689
* [bugfix] fix broken vision inference, change serialization for bytes by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/693
* Minor Quick Start documentation updates. by @derekslager in https://github.com/meta-llama/llama-stack/pull/692
* [bugfix] fix meta-reference agents w/ safety multiple model loading pytest by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/694
* [bugfix] fix prompt_adapter interleaved_content_convert_to_raw  by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/696
* Add missing "inline::" prefix for providers in building_distro.md by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/702
* Fix failing flake8 E226 check by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/701
* Add missing newlines before printing the Dockerfile content by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/700
* Add JSON structured outputs to Ollama Provider by @aidando73 in https://github.com/meta-llama/llama-stack/pull/680
* [#407] Agents: Avoid calling tools that haven't been explicitly enabled by @aidando73 in https://github.com/meta-llama/llama-stack/pull/637
* Made changes to readme and pinning to llamastack v0.0.61 by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/624
* [rag evals][1/n] refactor base scoring fn & data schema check by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/664
* [Post Training] Fix missing import by @SLR722 in https://github.com/meta-llama/llama-stack/pull/705
* Import from the right path  by @SLR722 in https://github.com/meta-llama/llama-stack/pull/708
* [#432] Add Groq Provider - chat completions by @aidando73 in https://github.com/meta-llama/llama-stack/pull/609
* Change post training run.yaml inference config  by @SLR722 in https://github.com/meta-llama/llama-stack/pull/710
* [Post training] make validation steps configurable by @SLR722 in https://github.com/meta-llama/llama-stack/pull/715
* Fix incorrect entrypoint for broken `llama stack run` by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/706
* Fix assert message and call to completion_request_to_prompt in remote:vllm by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/709
* Fix Groq invalid self.config reference by @aidando73 in https://github.com/meta-llama/llama-stack/pull/719
* support llama3.1 8B instruct in post training by @SLR722 in https://github.com/meta-llama/llama-stack/pull/698
* remove default logger handlers when using libcli with notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/718
* move DataSchemaValidatorMixin into standalone utils by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/720
* add 3.3 to together inference provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/729
* Update CODEOWNERS - add sixianyi0721 as the owner by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/731
* fix links for distro by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/733
* add --version to llama stack CLI & /version endpoint by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/732
* agents to use tools api by @dineshyv in https://github.com/meta-llama/llama-stack/pull/673
* Add X-LlamaStack-Client-Version, rename ProviderData -> Provider-Data by @ashwinb in https://github.com/meta-llama/llama-stack/pull/735
* Check version incompatibility by @ashwinb in https://github.com/meta-llama/llama-stack/pull/738
* Add persistence for localfs datasets by @VladOS95-cyber in https://github.com/meta-llama/llama-stack/pull/557
* Fixed typo in default VLLM_URL in remote-vllm.md by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/723
* Consolidating Memory tests under client-sdk by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/703
* Expose LLAMASTACK_PORT in cli.stack.run by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/722
* remove conflicting default for tool prompt format in chat completion by @dineshyv in https://github.com/meta-llama/llama-stack/pull/742
* rename LLAMASTACK_PORT to LLAMA_STACK_PORT for consistency with other env vars by @raghotham in https://github.com/meta-llama/llama-stack/pull/744
* Add inline vLLM inference provider to regression tests and fix regressions by @frreiss in https://github.com/meta-llama/llama-stack/pull/662
* [CICD] github workflow to push nightly package to testpypi by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/734
* Replaced zrangebylex method in the range method by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/521
* Improve model download doc by @SLR722 in https://github.com/meta-llama/llama-stack/pull/748
* Support building UBI9 base container image by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/676
* update notebook to use new tool defs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/745
* Add provider data passing for library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/750
* [Fireworks] Update model name for Fireworks by @benjibc in https://github.com/meta-llama/llama-stack/pull/753
* Consolidating Inference tests under client-sdk tests by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/751
* Consolidating Safety tests from various places under client-sdk by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/699
* [CI/CD] more robust re-try for downloading testpypi package by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/749
* [#432] Add Groq Provider - tool calls by @aidando73 in https://github.com/meta-llama/llama-stack/pull/630
* Rename ipython to tool by @ashwinb in https://github.com/meta-llama/llama-stack/pull/756
* Fix incorrect Python binary path for UBI9 image by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/757
* Update Cerebras docs to include header by @henrytwo in https://github.com/meta-llama/llama-stack/pull/704
* Add init files to post training folders by @SLR722 in https://github.com/meta-llama/llama-stack/pull/711
* Switch to use importlib instead of deprecated pkg_resources by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/678
* [bugfix] fix streaming GeneratorExit exception with LlamaStackAsLibraryClient by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/760
* Fix telemetry to work on reinstantiating new lib cli by @dineshyv in https://github.com/meta-llama/llama-stack/pull/761
* [post training]  define llama stack post training dataset format by @SLR722 in https://github.com/meta-llama/llama-stack/pull/717
* add braintrust to experimental-post-training template by @SLR722 in https://github.com/meta-llama/llama-stack/pull/763
* added support of PYPI_VERSION in stack build by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/762
* Fix broken tests in test_registry by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/707
* Fix fireworks run-with-safety template by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/766
* Free up memory after post training finishes by @SLR722 in https://github.com/meta-llama/llama-stack/pull/770
* Fix issue when generating distros by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/755
* Convert `SamplingParams.strategy` to a union by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/767
* [CICD] Github workflow for publishing Docker images by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/764
* [bugfix] fix llama guard parsing ContentDelta by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/772
* rebase eval test w/ tool_runtime fixtures by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/773
* More idiomatic REST API by @dineshyv in https://github.com/meta-llama/llama-stack/pull/765
* add nvidia distribution by @cdgamarose-nv in https://github.com/meta-llama/llama-stack/pull/565
* bug fixes on inference tests by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/774
* [bugfix] fix inference sdk test for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/775
* fix routing in library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/776
* [bugfix] fix client-sdk tests for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/777
* fix nvidia inference provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/781
* Make notebook testable by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/780
* Fix telemetry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/787
* fireworks add completion logprobs adapter by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/778
* Idiomatic REST API: Inspect by @dineshyv in https://github.com/meta-llama/llama-stack/pull/779
* Idiomatic REST API: Evals by @dineshyv in https://github.com/meta-llama/llama-stack/pull/782
* Add notebook testing to nightly build job by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/785
* [test automation] support run tests on config file  by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/730
* Idiomatic REST API: Telemetry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/786
* Make llama stack build not create a new conda by default by @ashwinb in https://github.com/meta-llama/llama-stack/pull/788
* REST API fixes by @dineshyv in https://github.com/meta-llama/llama-stack/pull/789
* fix cerebras template by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/790
* [Test automation] generate custom test report by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/739
* cerebras template update for memory by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/792
* Pin torchtune pkg version by @SLR722 in https://github.com/meta-llama/llama-stack/pull/791
* fix the code execution test in sdk tests by @dineshyv in https://github.com/meta-llama/llama-stack/pull/794
* add default toolgroups to all providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/795
* Fix tgi adapter by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/796
* Remove llama-guard in Cerebras template & improve agent test by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/798
* meta reference inference fixes by @ashwinb in https://github.com/meta-llama/llama-stack/pull/797
* fix provider model list test by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/800
* fix playground for v1 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/799
* fix eval notebook & add test to workflow by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/803
* add json_schema_type to ParamType deps by @dineshyv in https://github.com/meta-llama/llama-stack/pull/808
* Fixing small typo in quick start guide by @pmccarthy in https://github.com/meta-llama/llama-stack/pull/807
* cannot import name 'GreedySamplingStrategy' by @aidando73 in https://github.com/meta-llama/llama-stack/pull/806
* optional api dependencies by @ashwinb in https://github.com/meta-llama/llama-stack/pull/793
* fix vllm template by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/813
* More generic image type for OCI-compliant container technologies by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/802
* add mcp runtime as default to all providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/816
* fix vllm base64 image inference by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/815
* fix again vllm for non base64 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/818
* Fix incorrect RunConfigSettings due to the removal of conda_env by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/801
* Fix incorrect image type in publish-to-docker workflow by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/819
* test report for v0.1 by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/814
* [CICD] add simple test step for docker build workflow, fix prefix bug by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/821
* add section for mcp tool usage in notebook by @dineshyv in https://github.com/meta-llama/llama-stack/pull/831
* [ez] structured output for /completion ollama & enable tests by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/822
* add pytest option to generate a functional report for distribution by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/833
* bug fix for distro report generation by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/836
* [memory refactor][1/n] Rename Memory -> VectorIO, MemoryBanks -> VectorDBs by @ashwinb in https://github.com/meta-llama/llama-stack/pull/828
* [memory refactor][2/n] Update faiss and make it pass tests by @ashwinb in https://github.com/meta-llama/llama-stack/pull/830
* [memory refactor][3/n] Introduce RAGToolRuntime as a specialized sub-protocol by @ashwinb in https://github.com/meta-llama/llama-stack/pull/832
* [memory refactor][4/n] Update the client-sdk test for RAG by @ashwinb in https://github.com/meta-llama/llama-stack/pull/834
* [memory refactor][5/n] Migrate all vector_io providers by @ashwinb in https://github.com/meta-llama/llama-stack/pull/835
* [memory refactor][6/n] Update naming and routes by @ashwinb in https://github.com/meta-llama/llama-stack/pull/839
* Fix fireworks client sdk chat completion with images by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/840
* [inference api] modify content types so they follow a more standard structure by @ashwinb in https://github.com/meta-llama/llama-stack/pull/841

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

## What's Changed

A few important updates some of which are backwards incompatible. You must update your `run.yaml`s when upgrading. As always look to `templates/<distro>/run.yaml` for reference.

* Make embedding generation go through inference by @dineshyv in https://github.com/meta-llama/llama-stack/pull/606
* [/scoring] add ability to define aggregation functions for scoring functions & refactors by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/597
* Update the "InterleavedTextMedia" type  by @ashwinb in https://github.com/meta-llama/llama-stack/pull/635
* [NEW!] Experimental post-training APIs! https://github.com/meta-llama/llama-stack/pull/540,  https://github.com/meta-llama/llama-stack/pull/593, etc.

A variety of fixes and enhancements. Some selected ones:

* [#342] RAG - fix PDF format in vector database by @aidando73 in https://github.com/meta-llama/llama-stack/pull/551
* add completion api support to nvidia inference provider by @mattf in https://github.com/meta-llama/llama-stack/pull/533
* add model type to APIs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/588
* Allow using an "inline" version of Chroma using PersistentClient by @ashwinb in https://github.com/meta-llama/llama-stack/pull/567
* [docs] add playground ui docs by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/592
* add colab notebook & update docs by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/619
* [tests] add client-sdk pytests & delete client.py by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/638
* [bugfix] no shield_call when there's no shields configured by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/642

## New Contributors
* @SLR722 made their first contribution in https://github.com/meta-llama/llama-stack/pull/540
* @iamarunbrahma made their first contribution in https://github.com/meta-llama/llama-stack/pull/636

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.61...v0.0.62

---

# v0.0.61
Published on: 2024-12-10T20:50:33Z

## What's Changed
* add NVIDIA NIM inference adapter by @mattf in https://github.com/meta-llama/llama-stack/pull/355
* Tgi fixture by @dineshyv in https://github.com/meta-llama/llama-stack/pull/519
* fixes tests & move braintrust api_keys to request headers by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/535
* allow env NVIDIA_BASE_URL to set NVIDIAConfig.url by @mattf in https://github.com/meta-llama/llama-stack/pull/531
* move playground ui to llama-stack repo by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/536
* fix[documentation]: Update links to point to correct pages by @sablair in https://github.com/meta-llama/llama-stack/pull/549
* Fix URLs to Llama Stack Read the Docs Webpages by @JeffreyLind3 in https://github.com/meta-llama/llama-stack/pull/547
* Fix Zero to Hero README.md Formatting by @JeffreyLind3 in https://github.com/meta-llama/llama-stack/pull/546
* Guide readme fix by @raghotham in https://github.com/meta-llama/llama-stack/pull/552
* Fix broken Ollama link by @aidando73 in https://github.com/meta-llama/llama-stack/pull/554
* update client cli docs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/560
* reduce the accuracy requirements to pass the chat completion structured output test by @mattf in https://github.com/meta-llama/llama-stack/pull/522
* removed assertion in ollama.py and fixed typo in the readme by @wukaixingxp in https://github.com/meta-llama/llama-stack/pull/563
* Cerebras Inference Integration by @henrytwo in https://github.com/meta-llama/llama-stack/pull/265
* unregister API for dataset  by @sixianyi0721 in https://github.com/meta-llama/llama-stack/pull/507
* [llama stack ui] add native eval & inspect distro & playground pages by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/541
* Telemetry API redesign by @dineshyv in https://github.com/meta-llama/llama-stack/pull/525
* Introduce GitHub Actions Workflow for Llama Stack Tests by @ConnorHack in https://github.com/meta-llama/llama-stack/pull/523
* specify the client version that works for current together server by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/566
* remove unused telemetry related code by @dineshyv in https://github.com/meta-llama/llama-stack/pull/570
* Fix up safety client for versioned API by @stevegrubb in https://github.com/meta-llama/llama-stack/pull/573
* Add eval/scoring/datasetio API providers to distribution templates & UI developer guide by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/564
* Add ability to query and export spans to dataset by @dineshyv in https://github.com/meta-llama/llama-stack/pull/574
* Renames otel config from jaeger to otel by @codefromthecrypt in https://github.com/meta-llama/llama-stack/pull/569
* add telemetry docs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/572
* Console span processor improvements by @dineshyv in https://github.com/meta-llama/llama-stack/pull/577
* doc: quickstart guide errors by @aidando73 in https://github.com/meta-llama/llama-stack/pull/575
* Add kotlin docs by @Riandy in https://github.com/meta-llama/llama-stack/pull/568
* Update android_sdk.md by @Riandy in https://github.com/meta-llama/llama-stack/pull/578
* Bump kotlin docs to 0.0.54.1 by @Riandy in https://github.com/meta-llama/llama-stack/pull/579
* Make LlamaStackLibraryClient work correctly by @ashwinb in https://github.com/meta-llama/llama-stack/pull/581
* Update integration type for Cerebras to hosted by @henrytwo in https://github.com/meta-llama/llama-stack/pull/583
* Use customtool's get_tool_definition to remove duplication by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/584
* [#391] Add support for json structured output for vLLM by @aidando73 in https://github.com/meta-llama/llama-stack/pull/528
* Fix Jaeger instructions by @yurishkuro in https://github.com/meta-llama/llama-stack/pull/580
* fix telemetry import by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/585
* update template run.yaml to include openai api key for braintrust by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/590
* add tracing to library client by @dineshyv in https://github.com/meta-llama/llama-stack/pull/591
* Fixes for library client by @ashwinb in https://github.com/meta-llama/llama-stack/pull/587
* Fix issue 586 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/594

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

## What's Changed
* Fix TGI inference adapter
* Fix `llama stack build` in 0.0.54 by @dltn in https://github.com/meta-llama/llama-stack/pull/505
* Several documentation related improvements
* Fix opentelemetry adapter by @dineshyv in https://github.com/meta-llama/llama-stack/pull/510
* Update Ollama supported llama model list by @hickeyma in https://github.com/meta-llama/llama-stack/pull/483

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.54...v0.0.55

---

# v0.0.54
Published on: 2024-11-22T00:36:09Z

## What's Changed
* Bugfixes release on top of 0.0.53
* Don't depend on templates.py when print llama stack build messages by @ashwinb in https://github.com/meta-llama/llama-stack/pull/496
* Restructure docs by @dineshyv in https://github.com/meta-llama/llama-stack/pull/494
* Since we are pushing for HF repos, we should accept them in inference configs by @ashwinb in https://github.com/meta-llama/llama-stack/pull/497
* Fix fp8 quantization script. by @liyunlu0618 in https://github.com/meta-llama/llama-stack/pull/500
* use logging instead of prints by @dineshyv in https://github.com/meta-llama/llama-stack/pull/499

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

## What's Changed
* Update download command by @Wauplin in https://github.com/meta-llama/llama-stack/pull/9
* Update fbgemm version by @jianyuh in https://github.com/meta-llama/llama-stack/pull/12
* Add CLI reference docs by @dltn in https://github.com/meta-llama/llama-stack/pull/14
* Added Ollama as an inference impl  by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/20
* Hide older models by @dltn in https://github.com/meta-llama/llama-stack/pull/23
* Introduce Llama stack distributions by @ashwinb in https://github.com/meta-llama/llama-stack/pull/22
* Rename inline -> local by @dltn in https://github.com/meta-llama/llama-stack/pull/24
* Avoid using nearly double the memory needed by @ashwinb in https://github.com/meta-llama/llama-stack/pull/30
* Updates to prompt for tool calls by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/29
* RFC-0001-The-Llama-Stack by @raghotham in https://github.com/meta-llama/llama-stack/pull/8
* Add API keys to AgenticSystemConfig instead of relying on dotenv by @ashwinb in https://github.com/meta-llama/llama-stack/pull/33
* update cli ref doc by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/34
* fixed bug in download not enough disk space condition by @sisminnmaw in https://github.com/meta-llama/llama-stack/pull/35
* Updated cli instructions with additonal details for each subcommands by @varunfb in https://github.com/meta-llama/llama-stack/pull/36
* Updated URLs and addressed feedback by @varunfb in https://github.com/meta-llama/llama-stack/pull/37
* Fireworks basic integration by @benjibc in https://github.com/meta-llama/llama-stack/pull/39
* Together AI basic integration by @Nutlope in https://github.com/meta-llama/llama-stack/pull/43
* Update LICENSE by @raghotham in https://github.com/meta-llama/llama-stack/pull/47
* Add patch for SSE event endpoint responses by @dltn in https://github.com/meta-llama/llama-stack/pull/50
* API Updates: fleshing out RAG APIs, introduce "llama stack" CLI command by @ashwinb in https://github.com/meta-llama/llama-stack/pull/51
* [inference] Add a TGI adapter by @ashwinb in https://github.com/meta-llama/llama-stack/pull/52
* upgrade llama_models by @benjibc in https://github.com/meta-llama/llama-stack/pull/55
* Query generators for RAG query by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/54
* Add Chroma and PGVector adapters by @ashwinb in https://github.com/meta-llama/llama-stack/pull/56
* API spec update, client demo with Stainless SDK by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/58
* Enable Bing search by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/59
* add safety to openapi spec by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/62
* Add config file based CLI by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/60
* Simplified Telemetry API and tying it to logger by @ashwinb in https://github.com/meta-llama/llama-stack/pull/57
* [Inference] Use huggingface_hub inference client for TGI adapter by @hanouticelina in https://github.com/meta-llama/llama-stack/pull/53
* Support `data:` in URL for memory. Add ootb support for pdfs by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/67
* Remove request wrapper migration by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/64
* CLI Update: build -> configure -> run by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/69
* API Updates by @ashwinb in https://github.com/meta-llama/llama-stack/pull/73
* Unwrap ChatCompletionRequest for context_retriever  by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/75
* CLI - add back build wizard, configure with name instead of build.yaml by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/74
* CLI: add build templates support, move imports by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/77
* fix prompt with name args by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/80
* Fix memory URL parsing by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/81
* Allow TGI adaptor to have non-standard llama model names by @hardikjshah in https://github.com/meta-llama/llama-stack/pull/84
* [API Updates] Model / shield / memory-bank routing + agent persistence + support for private headers by @ashwinb in https://github.com/meta-llama/llama-stack/pull/92
* Bedrock Guardrails comiting after rebasing the fork by @rsgrewal-aws in https://github.com/meta-llama/llama-stack/pull/96
* Bedrock Inference Integration by @poegej in https://github.com/meta-llama/llama-stack/pull/94
* Support for Llama3.2 models and Swift SDK by @ashwinb in https://github.com/meta-llama/llama-stack/pull/98
* fix safety using inference by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/99
* Fixes typo for setup instruction for starting Llama Stack Server section  by @abhishekmishragithub in https://github.com/meta-llama/llama-stack/pull/103
* Make TGI adapter compatible with HF Inference API by @Wauplin in https://github.com/meta-llama/llama-stack/pull/97
* Fix links & format by @machina-source in https://github.com/meta-llama/llama-stack/pull/104
* docs: fix typo by @dijonkitchen in https://github.com/meta-llama/llama-stack/pull/107
* LG safety fix by @kplawiak in https://github.com/meta-llama/llama-stack/pull/108
* Minor typos, HuggingFace -> Hugging Face by @marklysze in https://github.com/meta-llama/llama-stack/pull/113
* Reordered pip install and llama model download by @KarthiDreamr in https://github.com/meta-llama/llama-stack/pull/112
* Update getting_started.ipynb by @delvingdeep in https://github.com/meta-llama/llama-stack/pull/117
* fix: 404 link to agentic system repository by @moldhouse in https://github.com/meta-llama/llama-stack/pull/118
* Fix broken links in RFC-0001-llama-stack.md by @bhimrazy in https://github.com/meta-llama/llama-stack/pull/134
* Validate `name` in `llama stack build` by @russellb in https://github.com/meta-llama/llama-stack/pull/128
* inference: Fix download command in error msg by @russellb in https://github.com/meta-llama/llama-stack/pull/133
* configure: Fix a error msg typo by @russellb in https://github.com/meta-llama/llama-stack/pull/131
* docs: Note how to use podman by @russellb in https://github.com/meta-llama/llama-stack/pull/130
* add env for LLAMA_STACK_CONFIG_DIR by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/137
* [bugfix] fix duplicate api endpoints by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/139
* Use inference APIs for executing Llama Guard by @ashwinb in https://github.com/meta-llama/llama-stack/pull/121
* fixing safety inference and safety adapter for new API spec. Pinned t… by @yogishbaliga in https://github.com/meta-llama/llama-stack/pull/105
* [CLI] remove dependency on CONDA_PREFIX in CLI by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/144
* [bugfix] fix #146 by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/147
* Extract provider data properly (attempt 2) by @ashwinb in https://github.com/meta-llama/llama-stack/pull/148
* `is_multimodal` accepts `core_model_id`  not model itself. by @wizardbc in https://github.com/meta-llama/llama-stack/pull/153
* fix broken bedrock inference provider by @moritalous in https://github.com/meta-llama/llama-stack/pull/151
* Fix podman+selinux compatibility by @russellb in https://github.com/meta-llama/llama-stack/pull/132
* docker: Install in editable mode for dev purposes by @russellb in https://github.com/meta-llama/llama-stack/pull/160
* [CLI] simplify docker run by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/159
* Add a RoutableProvider protocol, support for multiple routing keys by @ashwinb in https://github.com/meta-llama/llama-stack/pull/163
* docker: Check for selinux before using `--security-opt` by @russellb in https://github.com/meta-llama/llama-stack/pull/167
* Adds markdown-link-check and fixes a broken link by @codefromthecrypt in https://github.com/meta-llama/llama-stack/pull/165
* [bugfix] conda path lookup by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/179
* fix prompt guard by @ashwinb in https://github.com/meta-llama/llama-stack/pull/177
* inference: Add model option to client by @russellb in https://github.com/meta-llama/llama-stack/pull/170
* [CLI] avoid configure twice by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/171
* Check that the model is found before use. by @AshleyT3 in https://github.com/meta-llama/llama-stack/pull/182
* Add 'url' property to Redis KV config by @Minutis in https://github.com/meta-llama/llama-stack/pull/192
* Inline vLLM inference provider by @russellb in https://github.com/meta-llama/llama-stack/pull/181
* add databricks provider by @prithu-dasgupta in https://github.com/meta-llama/llama-stack/pull/83
* add Weaviate memory adapter by @zainhas in https://github.com/meta-llama/llama-stack/pull/95
* download: improve help text by @russellb in https://github.com/meta-llama/llama-stack/pull/204
* Fix ValueError in case chunks are empty by @Minutis in https://github.com/meta-llama/llama-stack/pull/206
* refactor docs by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/209
* README.md: Add vLLM to providers table by @russellb in https://github.com/meta-llama/llama-stack/pull/207
* Add .idea to .gitignore by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/216
* [bugfix] Fix logprobs on meta-reference impl by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/213
* Add classifiers in setup.py by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/217
* Add function for stopping inference by @kebbbnnn in https://github.com/meta-llama/llama-stack/pull/224
* JSON serialization for parallel processing queue by @dltn in https://github.com/meta-llama/llama-stack/pull/232
* Remove "routing_table" and "routing_key" concepts for the user by @ashwinb in https://github.com/meta-llama/llama-stack/pull/201
* ci: Run pre-commit checks in CI by @russellb in https://github.com/meta-llama/llama-stack/pull/176
* Fix incorrect completion() signature for Databricks provider by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/236
* Enable pre-commit on main branch by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/237
* Switch to pre-commit/action by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/239
* Remove request arg from chat completion response processing by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/240
* Fix broken rendering in Google Colab by @frntn in https://github.com/meta-llama/llama-stack/pull/247
* Docker compose scripts for remote adapters by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/241
* Update getting_started.md by @MeDott29 in https://github.com/meta-llama/llama-stack/pull/260
* Add llama download support for multiple models with comma-separated list by @tamdogood in https://github.com/meta-llama/llama-stack/pull/261
* config templates restructure, docs by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/262
* [bugfix] fix case for agent when memory bank registered without specifying provider_id by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/264
* Add an option to not use elastic agents for meta-reference inference by @ashwinb in https://github.com/meta-llama/llama-stack/pull/269
* Make all methods `async def` again; add completion() for meta-reference by @ashwinb in https://github.com/meta-llama/llama-stack/pull/270
* Add vLLM inference provider for OpenAI compatible vLLM server by @terrytangyuan in https://github.com/meta-llama/llama-stack/pull/178
* Update event_logger.py by @nehal-a2z in https://github.com/meta-llama/llama-stack/pull/275
* llama stack distributions / templates / docker refactor by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/266
* add more distro templates by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/279
* first version of readthedocs by @raghotham in https://github.com/meta-llama/llama-stack/pull/278
* add completion() for ollama by @dineshyv in https://github.com/meta-llama/llama-stack/pull/280
* [Evals API] [1/n] Initial API by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/287
* Add REST api example for chat_completion by @subramen in https://github.com/meta-llama/llama-stack/pull/286
* feat: Qdrant Vector index support by @Anush008 in https://github.com/meta-llama/llama-stack/pull/221
* Add support for Structured Output / Guided decoding by @ashwinb in https://github.com/meta-llama/llama-stack/pull/281
* [bug] Fix import conflict for SamplingParams by @subramen in https://github.com/meta-llama/llama-stack/pull/285
* Added implementations for get_agents_session, delete_agents_session and delete_agents by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/267
* [Evals API][2/n] datasets / datasetio meta-reference implementation by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/288
* Added tests for persistence by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/274
* Support structured output for Together by @ashwinb in https://github.com/meta-llama/llama-stack/pull/289
* dont set num_predict for all providers by @dineshyv in https://github.com/meta-llama/llama-stack/pull/294
* Fix issue w/ routing_table api getting added when router api is not specified by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/298
* New quantized models by @ashwinb in https://github.com/meta-llama/llama-stack/pull/301
* [Evals API][3/n] scoring_functions / scoring meta-reference implementations by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/296
* completion() for tgi by @dineshyv in https://github.com/meta-llama/llama-stack/pull/295
* [enhancement] added templates and enhanced readme by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/307
* Fix for get_agents_session by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/300
* fix broken --list-templates with adding build.yaml files for packaging by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/327
* Added hadamard transform for spinquant by @sacmehta in https://github.com/meta-llama/llama-stack/pull/326
* [Evals API][4/n] evals with generation meta-reference impl by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/303
* completion() for together by @dineshyv in https://github.com/meta-llama/llama-stack/pull/324
* completion() for fireworks by @dineshyv in https://github.com/meta-llama/llama-stack/pull/329
* [Evals API][6/n] meta-reference llm as judge, registration for ScoringFnDefs by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/330
* update distributions compose/readme by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/338
* distro readmes with model serving instructions by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/339
* [Evals API][7/n] braintrust scoring provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/333
* Kill --name from llama stack build by @ashwinb in https://github.com/meta-llama/llama-stack/pull/340
* Do not cache pip by @stevegrubb in https://github.com/meta-llama/llama-stack/pull/349
* add dynamic clients for all APIs by @ashwinb in https://github.com/meta-llama/llama-stack/pull/348
* fix bedrock impl by @dineshyv in https://github.com/meta-llama/llama-stack/pull/359
* [docs] update documentations by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/356
* pgvector fixes by @dineshyv in https://github.com/meta-llama/llama-stack/pull/369
* persist registered objects with distribution by @dineshyv in https://github.com/meta-llama/llama-stack/pull/354
* Significantly simpler and malleable test setup by @ashwinb in https://github.com/meta-llama/llama-stack/pull/360
* Correct a traceback in vllm by @stevegrubb in https://github.com/meta-llama/llama-stack/pull/366
* add postgres kvstoreimpl by @dineshyv in https://github.com/meta-llama/llama-stack/pull/374
* add ability to persist memory banks created for faiss by @dineshyv in https://github.com/meta-llama/llama-stack/pull/375
* fix postgres config validation by @dineshyv in https://github.com/meta-llama/llama-stack/pull/380
* Enable vision models for (Together, Fireworks, Meta-Reference, Ollama) by @ashwinb in https://github.com/meta-llama/llama-stack/pull/376
* Kill `llama stack configure` by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/371
* fix routing tables look up key for memory bank by @dineshyv in https://github.com/meta-llama/llama-stack/pull/383
* add bedrock distribution code by @dineshyv in https://github.com/meta-llama/llama-stack/pull/358
* Enable remote::vllm by @ashwinb in https://github.com/meta-llama/llama-stack/pull/384
* Directory rename: `providers/impls` -> `providers/inline`, `providers/adapters` -> `providers/remote` by @ashwinb in https://github.com/meta-llama/llama-stack/pull/381
* fix safety signature mismatch by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/388
* Remove the safety adapter for Together; we can just use "meta-reference" by @ashwinb in https://github.com/meta-llama/llama-stack/pull/387
* [LlamaStack][Fireworks] Update client and add unittest by @benjibc in https://github.com/meta-llama/llama-stack/pull/390
* [bugfix] fix together data validator by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/393
* Add provider deprecation support; change directory structure by @ashwinb in https://github.com/meta-llama/llama-stack/pull/397
* Factor out create_dist_registry by @dltn in https://github.com/meta-llama/llama-stack/pull/398
* [docs] refactor remote-hosted distro by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/402
* [Evals API][10/n] API updates for EvalTaskDef + new test migration by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/379
* Resource oriented design for shields by @dineshyv in https://github.com/meta-llama/llama-stack/pull/399
* Add pip install helper for test and direct scenarios by @dltn in https://github.com/meta-llama/llama-stack/pull/404
* migrate model to Resource and new registration signature by @dineshyv in https://github.com/meta-llama/llama-stack/pull/410
* [Docs] Zero-to-Hero notebooks and quick start documentation by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/368
* Distributions updates (slight updates to ollama, add inline-vllm and remote-vllm) by @ashwinb in https://github.com/meta-llama/llama-stack/pull/408
* added quickstart w ollama and toolcalling using together by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/413
* Split safety into (llama-guard, prompt-guard, code-scanner) by @ashwinb in https://github.com/meta-llama/llama-stack/pull/400
* fix duplicate `deploy` in  compose.yaml by @subramen in https://github.com/meta-llama/llama-stack/pull/417
* [Evals API][11/n] huggingface dataset provider + mmlu scoring fn by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/392
* Folder restructure for evals/datasets/scoring by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/419
* migrate memory banks to Resource and new registration by @dineshyv in https://github.com/meta-llama/llama-stack/pull/411
* migrate dataset to resource by @dineshyv in https://github.com/meta-llama/llama-stack/pull/420
* migrate evals to resource by @dineshyv in https://github.com/meta-llama/llama-stack/pull/421
* migrate scoring fns to resource by @dineshyv in https://github.com/meta-llama/llama-stack/pull/422
* Rename all inline providers with an inline:: prefix by @ashwinb in https://github.com/meta-llama/llama-stack/pull/423
* fix tests after registration migration & rename meta-reference -> basic / llm_as_judge provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/424
* fix eval task registration by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/426
* fix fireworks data validator by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/427
* Allow specifying resources in StackRunConfig by @ashwinb in https://github.com/meta-llama/llama-stack/pull/425
* Enable sane naming of registered objects with defaults by @ashwinb in https://github.com/meta-llama/llama-stack/pull/429
* Remove the "ShieldType" concept by @ashwinb in https://github.com/meta-llama/llama-stack/pull/430
* Inference to use provider resource id to register and validate by @dineshyv in https://github.com/meta-llama/llama-stack/pull/428
* Kill "remote" providers and fix testing with a remote stack properly by @ashwinb in https://github.com/meta-llama/llama-stack/pull/435
* add inline:: prefix for localfs provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/441
* change schema -> dataset_schema for Dataset class by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/442
* change schema -> dataset_schema for register_dataset api by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/443
* PR-437-Fixed bug to allow system instructions after first turn by @cheesecake100201 in https://github.com/meta-llama/llama-stack/pull/440
* add support for ${env.FOO_BAR} placeholders in run.yaml files by @ashwinb in https://github.com/meta-llama/llama-stack/pull/439
* model registration in ollama and vllm check against the available models in the provider by @dineshyv in https://github.com/meta-llama/llama-stack/pull/446
* Added link to the Colab notebook of the Llama Stack lesson on the Llama 3.2 course on DLAI by @jeffxtang in https://github.com/meta-llama/llama-stack/pull/445
* make distribution registry thread safe and other fixes by @dineshyv in https://github.com/meta-llama/llama-stack/pull/449
* local persistent for hf dataset provider by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/451
* Support model resource updates and deletes by @dineshyv in https://github.com/meta-llama/llama-stack/pull/452
* init registry once by @dineshyv in https://github.com/meta-llama/llama-stack/pull/450
* local persistence for eval tasks by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/453
* Fix build configure deprecation message by @hickeyma in https://github.com/meta-llama/llama-stack/pull/456
* Support parallel downloads for `llama model download` by @ashwinb in https://github.com/meta-llama/llama-stack/pull/448
* Add a verify-download command to llama CLI by @ashwinb in https://github.com/meta-llama/llama-stack/pull/457
* unregister for memory banks and remove update API by @dineshyv in https://github.com/meta-llama/llama-stack/pull/458
* move hf addapter->remote by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/459
* await initialize in faiss by @dineshyv in https://github.com/meta-llama/llama-stack/pull/463
* fix faiss serialize and serialize of index by @dineshyv in https://github.com/meta-llama/llama-stack/pull/464
* Extend shorthand support for the `llama stack run` command by @vladimirivic in https://github.com/meta-llama/llama-stack/pull/465
* [Agentic Eval] add ability to run agents generation by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/469
* Auto-generate distro yamls + docs  by @ashwinb in https://github.com/meta-llama/llama-stack/pull/468
* Allow models to be registered as long as llama model is provided by @dineshyv in https://github.com/meta-llama/llama-stack/pull/472
* get stack run config based on template name by @dineshyv in https://github.com/meta-llama/llama-stack/pull/477
* add quantized model ollama support by @wukaixingxp in https://github.com/meta-llama/llama-stack/pull/471
* Update kotlin client docs by @Riandy in https://github.com/meta-llama/llama-stack/pull/476
* remove pydantic namespace warnings using model_config by @mattf in https://github.com/meta-llama/llama-stack/pull/470
* fix llama stack build for together & llama stack build from templates by @yanxi0830 in https://github.com/meta-llama/llama-stack/pull/479
* Add version to REST API url by @ashwinb in https://github.com/meta-llama/llama-stack/pull/478
* support adding alias for models without hf repo/sku entry by @dineshyv in https://github.com/meta-llama/llama-stack/pull/481
* update quick start to have the working instruction by @chuenlok in https://github.com/meta-llama/llama-stack/pull/467
* add changelog by @dineshyv in https://github.com/meta-llama/llama-stack/pull/487
* Added optional md5 validate command once download is completed by @varunfb in https://github.com/meta-llama/llama-stack/pull/486
* Support Tavily as built-in search tool. by @iseeyuan in https://github.com/meta-llama/llama-stack/pull/485
* Reorganizing Zero to Hero Folder structure by @heyjustinai in https://github.com/meta-llama/llama-stack/pull/447
* fall to back to read from chroma/pgvector when not in cache by @dineshyv in https://github.com/meta-llama/llama-stack/pull/489
* register with provider even if present in stack by @dineshyv in https://github.com/meta-llama/llama-stack/pull/491
* Make run yaml optional so dockers can start with just --env by @ashwinb in https://github.com/meta-llama/llama-stack/pull/492

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
