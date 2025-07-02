# API Documentation Update Plan

## Objective
Systematically improve Llama Stack API reference documentation by adding comprehensive descriptions to unlabeled fields and endpoints in the auto-generated OpenAPI specification, with special focus on classes decorated with @json_schema_type and methods decorated with @webmethod.

## Task Overview

Update API documentation for all files in `/llama_stack/apis/` by:
1. **Priority Focus**: Adding comprehensive docstring documentation to @json_schema_type classes and @webmethod functions
2. Adding comprehensive docstring documentation using `:param field_name: description` syntax to BaseModel classes that lack them
3. Improving existing docstring documentation only if significantly lacking or incorrect
4. Ensuring consistency across identical field types throughout all APIs
5. Using actual code implementation as the source of truth for descriptions
6. Avoiding Field(description="...") code changes - use ONLY when absolutely necessary for complex technical fields

## Documentation Standards

### Docstring Style Guidelines

Based on analysis of the codebase (using `agents.py` as the gold standard), follow these conventions:

#### **Formatting Rules**
- **Capitalization**: Start all descriptions with uppercase letters
- **Punctuation**: No ending periods for single-sentence descriptions
- **Optional markers**: Use "(Optional)" prefix for optional parameters
- **Verb tense**: Present tense for field descriptions, imperative mood for method descriptions
- **Articles**: Use "the", "a", "an" appropriately for clarity

#### **Length Standards**
- **Class descriptions**: 1-2 sentences focusing on purpose and context
- **Parameter descriptions**: 1 sentence, occasionally 2 for complex parameters  
- **Return descriptions**: 1 sentence describing what is returned
- **Technical depth**: Appropriate for API documentation, avoid implementation details

### Description Standards
- **@json_schema_type classes**: High-level purpose, context, and complete `:param` documentation for all fields
- **@webmethod functions**: What it does, when to use it, important behaviors, parameter documentation, return value description
- **Field descriptions**: Clear, concise, explain purpose and constraints based on actual code usage  
- **Class descriptions**: High-level purpose and context
- **Method descriptions**: What it does, when to use it, important behaviors
- **Enum descriptions**: Explain each value's meaning and use case using `:cvar`
- **Consistency principle**: Identical field types should have identical descriptions across APIs
- **Conservation principle**: Preserve existing descriptions unless they are significantly lacking
- **Code-only information**: Base ALL descriptions on code context and commonly available knowledge - NEVER invent facts, papers, citations, or technical details

### Documentation Templates

#### Primary Pattern: ReST-style Docstrings (95% of codebase uses this)

**@json_schema_type Classes:**
```python
@json_schema_type
class ExampleModel(BaseModel):
    """Brief description of the model's purpose and context.

    :param field_name: Clear description of what this field represents, its purpose, and any constraints
    :param optional_field: (Optional) Description including when it's used and default behavior
    """
    
    field_name: str
    optional_field: str | None = None
```

**@webmethod Functions:**
```python
@webmethod(route="/example", method="POST", descriptive_name="create_example")
async def create_example(
    self,
    param1: str,
    param2: int | None = None,
) -> ExampleResponse:
    """Brief description of what the endpoint does and its purpose.

    :param param1: Description of the required parameter
    :param param2: (Optional) Description of the optional parameter and default behavior
    :returns: Description of the response object and what it contains
    """
```

**Enum Classes:**
```python
class ExampleEnum(StrEnum):
    """Brief description of the enum's purpose.
    
    :cvar value_one: Description of what this value represents
    :cvar value_two: Description of what this value represents
    """
    
    value_one = "value_one"
    value_two = "value_two"
```

#### Secondary Pattern: Field() Descriptions (use sparingly, only when absolutely necessary)
```python
# Use ONLY for complex technical fields requiring additional context:
complex_field: dict[str, Any] = Field(
    description="Complex technical description that cannot be adequately captured in class docstring"
)
```

### Parameter Type Documentation Examples

Based on codebase analysis, use these patterns for different parameter types:

#### **Simple Types**
```python
:param model_id: The identifier of the model to use
:param stream: Whether to stream the response
:param temperature: Controls randomness in the model's output
```

#### **Optional Types**
```python
:param documents: (Optional) List of documents to create the turn with
:param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False
:param max_tokens: (Optional) Maximum number of tokens to generate
```

#### **Complex Types**
```python
:param messages: List of messages to start the turn with
:param tool_calls: List of tool invocations requested by the model
:param sampling_params: Configuration parameters for model sampling behavior
```

#### **Union Types**
```python
:param content: The content of the attachment, which can include text and other media
:param input: Input message(s) to create the response, either as a string or list of structured inputs
```

#### **Dict Types**
```python
:param metadata: Set of 16 key-value pairs that can be attached to an object
:param params: Dictionary of parameters for the query operation
:param headers: Optional HTTP headers to include with the request
```

#### **Relationship Documentation**
```python
:param agent_id: The ID of the agent to create the turn for
:param session_id: The ID of the session to create the turn for
:param turn_id: The ID of the turn within the session
```

### Canonical Field Type Descriptions
Use these standard descriptions for consistency across APIs:

```python
# Core Identity Fields
:param agent_id: Unique identifier for the agent
:param session_id: Unique identifier for the conversation session  
:param turn_id: Unique identifier for the turn within a session
:param step_id: Unique identifier for the step within a turn

# Common Object References
:param completion_message: The model's generated response containing content and metadata
:param tool_calls: List of tool invocations requested by the model
:param safety_violation: Safety violation detected by content moderation, if any

# Temporal Fields
:param started_at: Timestamp when the operation began
:param completed_at: (Optional) Timestamp when the operation finished, if completed
:param created_at: Timestamp when the resource was created
```

### Quality Guidelines

#### **What Makes Good Documentation (based on agents.py patterns)**
- **Specific and actionable**: "List of documents to create the turn with" vs "Documents for the turn"
- **Context-aware**: "The ID of the agent to create the turn for" vs "Agent ID"
- **Constraint-inclusive**: "Limit can range between 1 and 100, and the default is 20"
- **Relationship-clear**: Shows how parameters relate to each other and the system
- **Behavior-descriptive**: Explains what happens when parameters are used

#### **Common Mistakes to Avoid**
- **Redundant descriptions**: Don't just repeat the parameter name
  - ❌ `:param agent_id: The agent ID`
  - ✅ `:param agent_id: Unique identifier for the agent`
- **Missing optional indicators**: Always mark optional parameters
  - ❌ `:param stream: Whether to stream the response`
  - ✅ `:param stream: (Optional) Whether to stream the response`
- **Inconsistent terminology**: Use the same terms for the same concepts
- **Implementation details**: Focus on usage, not internal implementation
- **Ending punctuation**: Don't add periods to single-sentence descriptions
- **Hallucinated information**: NEVER make up facts, papers, citations, or technical details not found in the code
  - ❌ `:param rrf_k: The impact factor for RRF scoring. Default of 60 is from the original RRF paper (Cormack et al., 2009)`
  - ✅ `:param rrf_k: The impact factor for RRF scoring. Higher values give more weight to higher-ranked results`

#### **Validation Checklist**
Before finalizing documentation, verify:
- [ ] All @json_schema_type classes have class docstrings
- [ ] All @webmethod functions have method docstrings  
- [ ] All parameters documented with :param
- [ ] All return values documented with :returns
- [ ] Optional parameters marked with (Optional)
- [ ] Consistent terminology used
- [ ] No ending periods on single sentences
- [ ] Descriptions start with uppercase letters
- [ ] No hallucinated facts, papers, citations, or invented technical details
- [ ] All information derives from code context or commonly available knowledge

## Implementation Process

### Execution Overview
**FOR EACH FILE in the processing order below:**
1. Read and analyze the file
2. Follow the 6-step process to update descriptions
3. Regenerate OpenAPI spec to verify changes
4. Move to next file

**COMPLETION CRITERIA:**
- All 22 files have been processed
- All @json_schema_type classes have comprehensive docstrings with complete `:param` field documentation
- All @webmethod functions have comprehensive docstrings with parameter and return documentation
- All other BaseModel classes have comprehensive docstrings with `:param` field documentation
- OpenAPI spec contains comprehensive descriptions for all fields and endpoints
- Existing descriptions are preserved unless significantly improved
- Minimal code changes - primarily docstring additions only

### File Processing Order (Complete Absolute Paths)
[x] 1. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/agents/agents.py` - Core agent system (start here, most complete)
[x] 2. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/inference/inference.py` - Core LLM functionality 
[x] 3. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/safety/safety.py` - Safety and moderation
[x] 4. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/models/models.py` - Model metadata and management
[x] 5. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/tools/tools.py` - Tool system APIs
[x] 6. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/tools/rag_tool.py` - RAG tool runtime
[x] 7. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/vector_io/vector_io.py` - Vector database operations
[x] 8. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/vector_dbs/vector_dbs.py` - Vector database management
[x] 9. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/files/files.py` - File management
[x] 10. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/datasets/datasets.py` - Dataset management
[x] 11. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/datasetio/datasetio.py` - Dataset I/O operations
[x] 12. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/post_training/post_training.py` - Training and fine-tuning
[x] 13. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/eval/eval.py` - Evaluation framework
[x] 14. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/scoring/scoring.py` - Scoring system
[x] 15. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/scoring_functions/scoring_functions.py` - Scoring function definitions
[x] 16. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/benchmarks/benchmarks.py` - Benchmarking framework
17. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/shields/shields.py` - Safety shields
18. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/batch_inference/batch_inference.py` - Batch inference operations
19. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/synthetic_data_generation/synthetic_data_generation.py` - Data generation
20. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/telemetry/telemetry.py` - Telemetry and monitoring
21. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/providers/providers.py` - Provider management
22. `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/inspect/inspect.py` - System inspection

**Step 1: Existing Documentation Assessment**
- **Priority**: Review all @json_schema_type classes and @webmethod functions for docstring completeness
- Review all existing class docstrings and `:param` documentation for quality and completeness
- Evaluate existing Field(description=...) annotations (minimal changes to these)
- Identify @json_schema_type classes and @webmethod functions completely missing docstrings vs. those with inadequate ones
- Document current description patterns for consistency reference

**Step 2: Documentation Audit and Prioritization**
- **Priority 1**: @json_schema_type classes with no docstrings at all
- **Priority 2**: @webmethod functions with no docstrings at all
- **Priority 3**: @json_schema_type classes with docstrings missing `:param` field documentation
- **Priority 4**: @webmethod functions with incomplete parameter/return documentation
- **Priority 5**: Other BaseModel classes with no docstrings at all
- **Priority 6**: Incomplete docstrings that need significant improvement
- **Priority 7**: Adequate docstrings that could use minor enhancements (be conservative here)
- Identify complex types that need explanation
- Check for deprecated fields that need deprecation notices

**Step 3: Deep Context Analysis**
- **Code Reading**: For each field/endpoint, read the actual implementation code to understand:
  - How the field is used in business logic
  - What values it can contain and their effects
  - Relationships with other fields and classes
  - Any validation rules or constraints
- **Cross-Reference Analysis**: When a field references another class (e.g., `CompletionMessage`):
  - Read the referenced class definition and its usage
  - Understand the full data flow and purpose
  - Follow method calls and implementations to understand behavior
- **Usage Pattern Review**: Check provider implementations, tests, and examples to see real-world usage
- **Dependency Tracking**: For complex types, understand their dependencies and relationships

**Step 4: Consistency Mapping**
- Create a master reference of common field types and their canonical descriptions
- Examples:
  - `agent_id: str` → "Unique identifier for the agent"
  - `CompletionMessage` → "The model's generated response containing content and metadata"
  - `session_id: str` → "Unique identifier for the conversation session"
- Ensure identical field types get consistent descriptions across all APIs
- Document any contextual variations where different descriptions are justified

**Step 5: Docstring Writing (Conservative Approach)**
- **For missing docstrings**: Write comprehensive class docstrings with complete `:param` documentation
- **For existing docstrings**: Only update if significantly lacking or incorrect
- **Preservation principle**: Keep existing docstring text, only add missing `:param` entries
- **Minimal code changes**: Avoid Field(description="...") modifications unless absolutely necessary
- Base all descriptions on actual code behavior, not assumptions from names
- Include technical details in docstrings like:
  - Expected data formats and constraints
  - When fields are populated vs. None
  - Relationships to other system components
  - Any behavioral implications

**Step 6: Verification and Testing**
- Regenerate OpenAPI spec: `uv run ./docs/openapi_generator/run_openapi_generator.sh`
- Compare new descriptions with existing ones to ensure improvements are justified
- Verify technical accuracy against actual code behavior
- Check consistency across similar fields in different APIs

### Step-by-Step Process for Each File

#### **Detailed Instructions for Each File**

1. **Use Read tool to examine the entire file**
   ```
   Read tool: file_path = "/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/agents/agents.py"
   ```
   (Replace "agents/agents.py" with the specific file from the processing list)

2. **Use Grep tool to find @json_schema_type classes**
   ```
   Grep tool: pattern = "@json_schema_type", path = "/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/agents/agents.py"
   ```

3. **Use Grep tool to find @webmethod functions**
   ```
   Grep tool: pattern = "@webmethod", path = "/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/agents/agents.py"
   ```

4. **For each @json_schema_type class found:**
   - Check if it has a class docstring
   - Check if all fields are documented with `:param field_name: description`
   - If missing, use Edit or MultiEdit tool to add comprehensive docstring

5. **For each @webmethod function found:**
   - Check if it has a method docstring
   - Check if all parameters are documented with `:param`
   - Check if return value is documented with `:returns`
   - If missing, use Edit or MultiEdit tool to add comprehensive docstring

6. **After completing file updates:**
   ```bash
   Bash tool: uv run ./docs/openapi_generator/run_openapi_generator.sh
   ```

#### **How to Identify Missing Documentation**

**For @json_schema_type classes:**
1. Use Grep tool to find all classes: `@json_schema_type`
2. For each result, use Read tool to check the lines immediately after
3. Look for this pattern (missing docstring):
   ```python
   @json_schema_type
   class ClassName(BaseModel):
       field_name: Type
   ```
4. If you see the above pattern WITHOUT a `"""docstring"""` between the class definition and first field, it needs documentation

**For @webmethod functions:**
1. Use Grep tool to find all methods: `@webmethod`
2. For each result, use Read tool to check the lines after the function definition
3. Look for this pattern (missing docstring):
   ```python
   @webmethod(route="/path", method="POST")
   async def method_name(self, param: Type) -> ReturnType:
       ...
   ```
4. If you see the above pattern WITHOUT a `"""docstring"""` between the function definition and `...`, it needs documentation

#### **Specific Tool Usage Patterns**

**For adding a new docstring to a class:**
```
Edit tool:
old_string: "@json_schema_type\nclass ClassName(BaseModel):"
new_string: "@json_schema_type\nclass ClassName(BaseModel):\n    \"\"\"Brief description.\n    \n    :param field1: Description of field1\n    :param field2: Description of field2\n    \"\"\""
```

**For adding missing :param entries to existing docstring:**
```
Edit tool:
old_string: "    \"\"\"Brief description.\n    \"\"\""
new_string: "    \"\"\"Brief description.\n    \n    :param field1: Description of field1\n    :param field2: Description of field2\n    \"\"\""
```

## Quality Assurance

### Description Quality Checklist
- [ ] **Code-Based**: Description derived from actual implementation, not assumptions
- [ ] **Accurate**: Description matches actual field behavior and usage patterns
- [ ] **Complete**: Covers purpose, constraints, format, and relationships
- [ ] **Clear**: Understandable to API users without ambiguity
- [ ] **Consistent**: Uses canonical descriptions for identical field types across APIs
- [ ] **Conservative**: Existing descriptions preserved unless significantly improved
- [ ] **Contextual**: Descriptions reflect how fields are actually used in the system
- [ ] **Helpful**: Provides actionable information for developers

### Special Cases to Address

**A. Union Types**
```python
# Document in class docstring:
class ExampleClass(BaseModel):
    """Class description.
    
    :param content: Either interleaved content (text/images) or a URL reference
    """
    content: InterleavedContent | URL
```

**B. List Types**
```python
# Document in class docstring:
class ExampleClass(BaseModel):
    """Class description.
    
    :param steps: Ordered list of processing steps executed during this turn
    """
    steps: list[Step]
```

**C. Optional Fields with Defaults**
```python
# Document in class docstring, mention optional nature:
class ExampleClass(BaseModel):
    """Class description.
    
    :param output_attachments: (Optional) Files or media attached to the agent's response
    """
    output_attachments: list[Attachment] | None = Field(default_factory=lambda: [])
```

**D. Deprecated Fields (preserve existing Field() usage)**
```python
# Only modify if Field() already exists with deprecated parameter:
class ExampleClass(BaseModel):
    """Class description.
    
    :param tool_choice: (Deprecated) Whether tool use is automatic, required, or none
    """
    tool_choice: ToolChoice | None = Field(
        default=None,
        deprecated="Use tool_config.tool_choice instead"
    )
```


## Tools and Commands Reference

### Essential Commands for Each File
```bash
# Regenerate OpenAPI documentation after changes
uv run ./docs/openapi_generator/run_openapi_generator.sh
```

### Useful Search Patterns
```bash
# Find @json_schema_type classes - use via Grep tool
grep -r "@json_schema_type" <file_path>

# Find @webmethod functions - use via Grep tool
grep -r "@webmethod" <file_path>

# Find fields without Field(description=...) - use via Grep tool
grep -r "^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*:[[:space:]]*.*[^)]$" <file_path> | grep -v "Field("

# Find existing descriptions for consistency - use via Grep tool  
grep -r "Field(" <file_path> | grep "description="

# Find class usage patterns - use via Grep tool
grep -r "ClassName" llama_stack/providers/
```

### Tool Usage Instructions
- **Read tool**: Use to examine files completely
- **Grep tool**: Use to search for patterns within files/directories
- **Edit tool**: Use to make single field description changes
- **MultiEdit tool**: Use to make multiple changes to the same file
- **Bash tool**: Use to regenerate OpenAPI specs and run validation

## Critical Implementation Guidelines

### 1. Code-First Approach
- **ALWAYS** read the actual implementation code before writing descriptions
- Trace through method calls and class relationships to understand full context
- Use provider implementations and tests as additional context sources
- Never guess based on variable names alone

### 2. Conservation Principle  
- **PRESERVE** existing descriptions unless they are significantly lacking or incorrect
- When improving existing descriptions, keep original text and only add necessary clarifications
- Document why any existing description was changed (should be rare)

### 3. Consistency Enforcement
- Maintain a master reference of canonical descriptions for common field types
- Use identical descriptions for identical field types across all APIs
- Only vary descriptions when there's a meaningful contextual difference

### 4. Quality Gates
- Before any description update, verify it against actual code behavior
- Ensure the description helps developers understand the field's purpose and usage
- Test descriptions by checking if they match the OpenAPI output expectations
- **CRITICAL**: Verify that NO information is invented - all descriptions must derive from code context or commonly available knowledge
- Never include academic papers, citations, or technical details not found in the codebase

### 5. Documentation Implementation Rules

#### **ONLY Add Docstrings - Do NOT Modify Code**
- **Primary approach**: Add or improve class docstrings with `:param` documentation
- **Avoid Field() changes**: Do NOT add `Field(description="...")` unless the Field() already exists for other reasons
- **Preserve existing Field()**: If Field() already exists with description, preserve it and also add docstring documentation

#### **Specific Decision Tree**
```
For each @json_schema_type class:
1. Does it have a class docstring? 
   - NO → Add comprehensive docstring with all :param entries
   - YES → Check if all fields are documented in docstring
     - Missing fields → Add missing :param entries to existing docstring

For each field in the class:
2. Does the field use Field() with existing description?
   - YES → Preserve the Field(), also ensure field is documented in class docstring
   - NO → Document ONLY in class docstring, do NOT add Field(description="...")

For each @webmethod function:
3. Does it have a method docstring?
   - NO → Add comprehensive docstring with :param and :returns
   - YES → Check completeness and add missing parts
```

#### **Never Modify These Code Elements**
- Do NOT add new Field() annotations 
- Do NOT modify existing Field() parameters (except in very rare cases)
- Do NOT change type annotations
- Do NOT modify function signatures
- Do NOT change import statements

#### **Practical Example from agents.py**

**Current code (missing docstring) - Lines 233-237:**
```python
@json_schema_type
class Agent(BaseModel):
    agent_id: str
    agent_config: AgentConfig
    created_at: datetime
```

**After applying guidelines (add docstring only):**
```python
@json_schema_type
class Agent(BaseModel):
    """An agent instance with configuration and metadata.

    :param agent_id: Unique identifier for the agent
    :param agent_config: Configuration settings for the agent
    :param created_at: Timestamp when the agent was created
    """
    agent_id: str
    agent_config: AgentConfig
    created_at: datetime
```

**Exact Tool usage for this change:**
```
Edit tool:
file_path: "/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/agents/agents.py"
old_string: "@json_schema_type\nclass Agent(BaseModel):\n    agent_id: str"
new_string: "@json_schema_type\nclass Agent(BaseModel):\n    \"\"\"An agent instance with configuration and metadata.\n\n    :param agent_id: Unique identifier for the agent\n    :param agent_config: Configuration settings for the agent\n    :param created_at: Timestamp when the agent was created\n    \"\"\"\n    agent_id: str"
```

#### **Step-by-Step Validation Process**

After making changes to any file:

1. **Verify syntax** - Use Read tool to check the file reads correctly
2. **Check @json_schema_type coverage** - Use Grep to find all instances and verify each has docstring
3. **Check @webmethod coverage** - Use Grep to find all instances and verify each has docstring  
4. **Test OpenAPI generation** - Use Bash tool: `uv run ./docs/openapi_generator/run_openapi_generator.sh`
5. **Validate output** - Use Read tool to check `docs/_static/llama-stack-spec.yaml` contains new descriptions

### 6. Common Scenarios and Troubleshooting

#### **Classes That Don't Need @json_schema_type**
- Some BaseModel classes in the codebase are not marked with @json_schema_type
- These are typically internal/utility classes not exposed in the API
- Still add docstrings to these for completeness, but they are lower priority

#### **Complex Inheritance Patterns**
- Some classes inherit from others (e.g., `InferenceStep(StepCommon)`)
- Document inherited fields in the subclass docstring as well
- Don't assume parent class documentation is sufficient

#### **When Field() Already Exists**
```python
# Current code with existing Field():
class Example(BaseModel):
    field_name: str = Field(default="value", description="Existing description")

# Add docstring while preserving Field():
class Example(BaseModel):
    """Class description.
    
    :param field_name: Existing description
    """
    field_name: str = Field(default="value", description="Existing description")
```

#### **Protocol Classes**
- Some files contain Protocol classes (like `class Agents(Protocol):`)
- Focus on @webmethod functions within these protocols
- Protocol class itself may not need @json_schema_type documentation

#### **File Processing Validation**
After each file, verify:
1. All @json_schema_type classes have docstrings
2. All @webmethod functions have docstrings  
3. No new Field() annotations were added unnecessarily
4. No syntax errors introduced
5. Existing Field() descriptions preserved

## Final Validation

After processing all 22 files:
1. Use `Bash` tool to regenerate final OpenAPI spec
2. Use `Read` tool to verify the complete `docs/_static/llama-stack-spec.yaml` 
3. Confirm all API endpoints and fields have descriptions
4. Check that the documentation is comprehensive and consistent

## Success Indicators
- [ ] All 22 API files processed
- [ ] All @json_schema_type classes have comprehensive docstrings with `:param` field documentation
- [ ] All @webmethod functions have comprehensive docstrings with parameter and return documentation
- [ ] OpenAPI spec contains comprehensive field descriptions
- [ ] Existing descriptions preserved unless significantly improved
- [ ] Consistent descriptions across identical field types
- [ ] All descriptions based on actual code behavior

## Test Case Verification

To verify the plan works correctly, test against `/Users/saip/Documents/GitHub/llama-stack/llama_stack/apis/safety/safety.py`:

**Expected findings:**
- 3 @json_schema_type classes without docstrings: `ViolationLevel`, `SafetyViolation`, `RunShieldResponse`
- 1 @webmethod function with good docstring: `run_shield` (preserve as-is)

**Expected actions:**
1. Add enum docstring to `ViolationLevel` with `:cvar` entries
2. Add class docstring to `SafetyViolation` with `:param` entries  
3. Add class docstring to `RunShieldResponse` with `:param` entry
4. Preserve existing `run_shield` docstring (already complete)
5. Do NOT modify the existing `Field(default_factory=dict)` in SafetyViolation

**Expected result:**
All classes documented, no code modifications, OpenAPI spec improved.

This systematic approach ensures comprehensive, accurate, and maintainable API documentation that will significantly improve the developer experience when working with Llama Stack APIs.