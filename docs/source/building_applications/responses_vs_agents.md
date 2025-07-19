# Agents vs OpenAI Responses API

Llama Stack provides two different APIs for building AI applications with tool calling capabilities: the **Agents API** and the **OpenAI Responses API**. While both enable AI systems to use tools, and maintain full conversation history, they serve different use cases and have distinct characteristics.

## Overview

### Agents API
The Agents API is a full-featured, stateful system designed for complex, multi-turn conversations with persistent sessions. It provides comprehensive agent lifecycle management, detailed execution tracking, and rich metadata about each interaction through a structured session/turn/step hierarchy. The API can orchestrate multiple tool calls within a single turn, automatically handling all intermediate steps.

### OpenAI Responses API
The OpenAI Responses API is a lightweight system that provides a simple interface for single-turn tool calling. It's designed to be directly compatible with OpenAI's API patterns while adding Llama Stack's tool calling capabilities. Each response can call one tool at a time, requiring manual chaining for complex workflows. The API maintains conversation context through response chaining via `previous_response_id` and allows branching from any previous response point.

## Key Differences

| Feature | Agents API | OpenAI Responses API |
|---------|------------|---------------------|
| **State Management** | Hierarchical: Sessions → Turns → Steps | Linear: Chained responses |
| **Context Handling** | Session-based with persistent turns | Response chaining via `previous_response_id` |
| **Functional Scope** | Complete agent lifecycle management | Simple request/response pattern |
| **Execution Tracking** | Detailed step-by-step execution logs | Basic response with tool calls |
| **Tool Calling** | Can orchestrate multiple tool calls in a single turn | One tool call per response |
| **Conversation Branching** | Session-based conversation flow | Can branch from any previous response ID |
| **OpenAI Compatibility** | Proprietary API schema | Directly compatible with OpenAI client libraries |
| **Safety Features** | Built-in safety shields and custom safety models | No customizable safety features |

## Use Case Example comparing Agents and Responses APIs: Customer Support Ticket Resolution

Let's compare how both APIs handle a complex customer support scenario where an agent needs to:
1. Retrieve customer information
2. Check account status
3. Verify recent transactions
4. Generate a support ticket
5. Send confirmation email

### Agents API: Multiple tools in a single turn

```python
# Create agent with safety shields
agent_config = Agent(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input_shields=["meta-llama/Llama-Guard-3-8B"],
    output_shields=["meta-llama/Llama-Guard-3-8B"],
    tools=[
        get_customer_info,
        check_account_status,
        verify_transactions,
        create_ticket,
        send_email,
    ],
)

# Single turn handles the entire workflow
session_id = agent.create_session("customer_support")
response = agent.create_turn(
    session_id=session_id,
    messages=[
        {
            "role": "user",
            "content": "Customer ID 12345 is reporting unauthorized charges. Please investigate and create a support ticket.",
        }
    ],
    session_id=session_id,
)
for log in AgentEventLogger().log(response):
    log.print()

# The agent automatically:
# 1. Calls get_customer_info to retrieve customer details
# 2. Calls check_account_status to verify account state
# 3. Calls verify_transactions to check recent activity
# 4. Calls create_ticket to generate support ticket
# 5. Calls send_email to notify customer
# 6. Provides a comprehensive response

# All steps are tracked and can be reviewed
pprint(f"Input: {response.input_messages}")
pprint(f"Output: {response.output_message.content}")
pprint(f"Steps: {response.steps}")
```

### Responses API: Multiple Manual Responses Required

```python
# First response: Get customer information
response1 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Customer ID 12345 is reporting unauthorized charges. Get customer information.",
        }
    ],
    tools=[get_customer_info],
)

# Second response: Check account status
response2 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Check account status for the customer from previous response.",
        }
    ],
    tools=[check_account_status],
    previous_response_id=response1.id,
)

# Third response: Verify transactions
response3 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Verify recent transactions for the customer from previous responses.",
        }
    ],
    tools=[verify_transactions],
    previous_response_id=response2.id,
)

# Fourth response: Create support ticket
response4 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Create a support ticket based on all previous investigation results.",
        }
    ],
    tools=[create_ticket],
    previous_response_id=response3.id,
)

# Fifth response: Send confirmation email
response5 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Send confirmation email to customer about the support ticket.",
        }
    ],
    tools=[send_email],
    previous_response_id=response4.id,
)

# Sixth response: Provide final summary
response6 = responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[
        {
            "role": "user",
            "content": "Provide a comprehensive summary of all actions taken.",
        }
    ],
    previous_response_id=response5.id,
)
```

## Use Case Examples

### When to Use the Agents API

#### 1. **Customer Support Chatbot**
You're building a customer support system that needs to handle complex multi-step workflows.

**Scenario**: A customer contacts support about a login issue. The agent needs to execute multiple steps such as check account status, verify credentials, reset password, and send confirmation email.

**Why Agents API?** The agent can coordinate each step in the account recovery process, automatically handling the order of operations and sharing information between steps.

#### 2. **Research Assistant**
You need an AI assistant that can conduct multi-step research and analysis.

**Scenario**: You're researching AI safety regulations. The assistant needs to search multiple databases, analyze documents, cross-reference findings, and generate a comprehensive report.

**Why Agents API?** The agent can automatically orchestrate multiple tool calls (database searches, document analysis, cross-referencing) in sequence within a single turn, maintaining context throughout the entire research process.

#### 3. **Financial Compliance Review**
A financial firm needs to automate compliance checks for new client onboarding.

**Scenario**: The agent must verify customer identity, cross-check against sanction lists, validate submitted documents, calculate risk scores, and generate compliance reports, ensuring each action is logged for regulatory purposes.

**Why Agents API?** The agent can orchestrate the entire compliance workflow in one turn, automatically calling tools in the correct sequence while maintaining detailed execution logs for auditing and troubleshooting.

#### 4. **Medical Triage Assistant**
A healthcare provider wants an AI agent to help triage patient cases.

**Scenario**: A patient describes symptoms through chat. The agent must collect patient information, look up medical guidelines, suggest possible causes, recommend next steps, and schedule a follow-up appointment, while tracking each step of the interaction.

**Why Agents API?** The agent can coordinate all necessary tool calls in one orchestrated workflow, maintain patient context throughout the interaction, and provide an auditable record of each decision and recommendation.

### When to Use the OpenAI Responses API

#### 1. **OpenAI-Compatible Integrations**
You're building an application that needs to be compatible with existing OpenAI client libraries.

**Scenario**: You have an existing application using OpenAI's API and want to switch to Llama Stack without changing your client code.

**Why OpenAI Responses API?** The Responses API is fully compatible with OpenAI's API patterns, making it easy to switch between OpenAI endpoints and Llama Stack endpoints with minimal code changes.

#### 2. **Simple Tool Integration**
You want to add tool calling to an existing application with minimal complexity.

**Scenario**: You have a weather app that needs to get current weather information.

**Why OpenAI Responses API?** Simple tool-calling without having to manage sessions, turns, or complex workflow state. Each request is independent, but you can maintain context by chaining responses.

#### 3. **Rapid Prototyping and Testing**
You're quickly prototyping a tool integration or testing different models.

**Scenario**: You're building a proof-of-concept for automated email summarization. You want to test how different models or prompt templates perform on the same set of emails, and also maintain conversation context across multiple related messages. Additionally, you'd like to experiment by issuing new queries from any point in the conversation, using a specific previous response as the starting context.

**Why OpenAI Responses API?** The ability to branch from any previous response ID makes it easy to chain responses and maintain conversation context across related emails. More importantly, you can re-query from any previous response, enabling side-by-side testing, branching experiments, and "what if" scenarios.

#### 4. **Batch Processing with Context**
You need to process multiple related requests while maintaining context between them without the overhead of session management.

**Scenario**: You have a CSV file of product descriptions and want to generate marketing taglines for each product. To ensure consistency within a product line or campaign, you want each new tagline to build on context from previous products in the same category. Sometimes, you may also want to revisit a specific point and generate alternative taglines from there.

**Why OpenAI Responses API?** This lets you easily chain requests so each tagline can reference earlier responses for category or campaign consistency. If you want to branch off and generate alternative taglines for a specific product, you can re-query using the relevant response ID.

## For More Information

- **Agents API**: For detailed information on creating and managing agents, see the [Agents documentation](agent.md)
- **OpenAI Responses API**: For information on using the OpenAI-compatible responses API, see the [OpenAI API documentation](../openai/index.md)
