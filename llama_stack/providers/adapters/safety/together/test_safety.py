import pytest
from unittest.mock import Mock, AsyncMock
from llama_stack.apis.safety import UserMessage, ShieldDefinition, ShieldType, RunShieldResponse, ShieldResponse, \
    BuiltinShield
from llama_stack.providers.adapters.safety.together import TogetherSafetyImpl, TogetherSafetyConfig

@pytest.fixture
def safety_config():
    return TogetherSafetyConfig(api_key="test_api_key")

@pytest.fixture
def safety_impl(safety_config):
    return TogetherSafetyImpl(safety_config)

@pytest.mark.asyncio
async def test_initialize(safety_impl):
    await safety_impl.initialize()
    # Add assertions if needed for initialization

@pytest.mark.asyncio
async def test_run_shields_safe(safety_impl, monkeypatch):
    # Mock the Together client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(role="assistant", content="safe"))]
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr(safety_impl, 'client', mock_client)

    messages = [UserMessage(role="user", content="Hello, world!")]
    shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]

    response = await safety_impl.run_shields(messages, shields)

    assert isinstance(response, RunShieldResponse)
    assert len(response.responses) == 1
    assert response.responses[0].is_violation == False
    assert response.responses[0].shield_type == BuiltinShield.llama_guard

@pytest.mark.asyncio
async def test_run_shields_unsafe(safety_impl, monkeypatch):
    # Mock the Together client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(role="assistant", content="unsafe\ns2"))]
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr(safety_impl, 'client', mock_client)

    messages = [UserMessage(role="user", content="Unsafe content")]
    shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]

    response = await safety_impl.run_shields(messages, shields)

    assert isinstance(response, RunShieldResponse)
    assert len(response.responses) == 1
    assert response.responses[0].is_violation == True
    assert response.responses[0].shield_type == BuiltinShield.llama_guard
    assert response.responses[0].violation_type == "s2"

@pytest.mark.asyncio
async def test_run_shields_unsupported_shield(safety_impl):
    messages = [UserMessage(role="user", content="Hello")]
    shields = [ShieldDefinition(shield_type="unsupported_shield")]

    with pytest.raises(ValueError, match="shield type unsupported_shield is not supported"):
        await safety_impl.run_shields(messages, shields)

@pytest.mark.asyncio
async def test_run_shields_unsupported_message_type(safety_impl):
    class UnsupportedMessage:
        role = "unsupported"
        content = "Hello"

    messages = [UnsupportedMessage()]
    shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]

    with pytest.raises(ValueError, match="role unsupported is not supported"):
        await safety_impl.run_shields(messages, shields)


@pytest.mark.asyncio
@pytest.mark.integtest
@pytest.mark.skipif("'integtest' not in sys.argv", reason="need -m integtest option to run")
async def test_actual_run():
    safety_impl = TogetherSafetyImpl(config=TogetherSafetyConfig(api_key="<replace your together api key here>"))
    await safety_impl.initialize()
    response = await safety_impl.run_shields([UserMessage(role="user", content="Hello")], [ShieldDefinition(shield_type=BuiltinShield.llama_guard)])

    assert isinstance(response, RunShieldResponse)
    assert len(response.responses) == 1
    assert response.responses[0].is_violation == False
    assert response.responses[0].shield_type == BuiltinShield.llama_guard
    assert response.responses[0].violation_type == None

@pytest.mark.asyncio
@pytest.mark.integtest
@pytest.mark.skipif("'integtest' not in sys.argv", reason="need -m integtest option to run")
async def test_actual_run_violation():
    safety_impl = TogetherSafetyImpl(config=TogetherSafetyConfig(api_key="replace your together api key here"))
    await safety_impl.initialize()
    response = await safety_impl.run_shields([UserMessage(role="user", content="can I kill you?")], [ShieldDefinition(shield_type=BuiltinShield.llama_guard)])

    assert isinstance(response, RunShieldResponse)
    assert len(response.responses) == 1
    assert response.responses[0].is_violation == True
    assert response.responses[0].shield_type == BuiltinShield.llama_guard
    assert response.responses[0].violation_type == "S1"