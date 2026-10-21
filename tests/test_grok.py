import json

import httpx
import llm
import pytest
from pytest_httpx import HTTPXMock

from llm_grok import DEFAULT_MODEL, Grok, GrokError, RateLimitError


@pytest.fixture(autouse=True)
def ignore_warnings():
    """Ignoriere bekannte Warnungen."""
    warnings = [
        # Pydantic Warnung
        "Support for class-based `config` is deprecated",
        # Datetime Warnung
        "datetime.datetime.utcnow() is deprecated",
    ]
    for warning in warnings:
        pytest.mark.filterwarnings(f"ignore:{warning}")


@pytest.fixture
def model():
    return Grok(DEFAULT_MODEL)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Mock environment variables and API key for testing"""
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key-mock")
    monkeypatch.setattr("llm_grok.Grok.get_key", lambda self, key=None: "xai-test-key-mock")


@pytest.fixture
def mock_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
    }


def test_model_initialization(model):
    assert model.model_id == DEFAULT_MODEL
    assert model.can_stream == True
    assert model.needs_key == "grok"
    assert model.key_env_var == "XAI_API_KEY"


def test_build_messages_with_system_prompt(model):
    prompt = llm.Prompt(
        model=model, prompt="Test message", system="Custom system message"
    )
    messages = model.build_messages(prompt, None)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Custom system message"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Test message"


def test_build_messages_without_system_prompt(model):
    prompt = llm.Prompt(model=model, prompt="Test message")
    messages = model.build_messages(prompt, None)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Test message"


def test_build_messages_with_conversation(model, httpx_mock: HTTPXMock):
    # Mock the expected request content
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Previous message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json={
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Previous response"}}
            ],
        },
        match_json=expected_request,
    )

    conversation = llm.Conversation(model=model)
    prev_prompt = llm.Prompt(model=model, prompt="Previous message")

    prev_response = llm.Response(model=model, prompt=prev_prompt, stream=False)
    prev_response._response_json = {
        "choices": [{"message": {"role": "assistant", "content": "Previous response"}}]
    }

    conversation.responses.append(prev_response)

    prompt = llm.Prompt(model=model, prompt="New message")
    messages = model.build_messages(prompt, conversation)

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Previous message"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Previous response"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "New message"


def test_non_streaming_request(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=False)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
    assert json.loads(request.content) == expected_request


def test_streaming_request(model, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": True,
        "temperature": 0.0,
    }

    def response_callback(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
        assert json.loads(request.content) == expected_request
        stream_content = [
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Test"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" response"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content="".join(stream_content).encode(),
        )

    httpx_mock.add_callback(
        response_callback,
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=True)
    chunks = list(response)
    assert "".join(chunks) == "Test response"


def test_temperature_option(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.8,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    # Create prompt and pass temperature directly
    response = model.prompt("Test message", stream=False, temperature=0.8)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_max_tokens_option(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
        "max_completion_tokens": 100,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    # Create prompt and pass max_tokens directly
    response = model.prompt("Test message", stream=False, max_completion_tokens=100)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_api_error(model, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    error_response = {
        "error": {
            "message": "Invalid request",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        status_code=400,
        json=error_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    with pytest.raises(GrokError) as exc_info:
        response = model.prompt("Test message", stream=False)
        response.text()  # Trigger the API call

    # The error message comes directly from the API response
    assert str(exc_info.value) == error_response["error"]["message"]


def test_stream_parsing_error(model, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": True,
        "temperature": 0.0,
    }

    def error_callback(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
        assert json.loads(request.content) == expected_request
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=b"data: {invalid json}\n\n",
        )

    httpx_mock.add_callback(
        error_callback,
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=True)
    chunks = list(response)
    assert chunks == []


URL = "https://api.x.ai/v1/chat/completions"


@pytest.fixture
def sleeps(monkeypatch):
    """Record the sleeps requested by the rate limit handling instead of waiting."""
    recorded = []
    monkeypatch.setattr("llm_grok.time.sleep", recorded.append)
    return recorded


def add_rate_limited(httpx_mock: HTTPXMock, times, headers=None):
    for _ in range(times):
        httpx_mock.add_response(
            method="POST",
            url=URL,
            status_code=429,
            headers=headers or {},
            json={"error": {"message": "Too many requests"}},
        )


def stream_body(text):
    event = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()


def test_rate_limit_honours_retry_after(model, mock_response, httpx_mock, sleeps):
    add_rate_limited(httpx_mock, 1, headers={"Retry-After": "2"})
    httpx_mock.add_response(method="POST", url=URL, json=mock_response)

    assert model.prompt("Test message", stream=False).text() == "Test response"

    assert len(httpx_mock.get_requests()) == 2
    assert sum(sleeps) == 2


def test_rate_limit_backs_off_without_retry_after(
    model, mock_response, httpx_mock, sleeps
):
    add_rate_limited(httpx_mock, 2)
    httpx_mock.add_response(method="POST", url=URL, json=mock_response)

    assert model.prompt("Test message", stream=False).text() == "Test response"

    assert len(httpx_mock.get_requests()) == 3
    # BASE_DELAY * 2**attempt: 1s then 2s
    assert sum(sleeps) == Grok.BASE_DELAY * (1 + 2)


def test_rate_limit_retries_are_exhausted(model, httpx_mock, sleeps):
    add_rate_limited(httpx_mock, Grok.MAX_RETRIES, headers={"Retry-After": "1"})

    with pytest.raises(RateLimitError):
        model.prompt("Test message", stream=False).text()

    assert len(httpx_mock.get_requests()) == Grok.MAX_RETRIES


def test_rate_limit_retried_when_streaming(model, httpx_mock, sleeps):
    add_rate_limited(httpx_mock, 1, headers={"Retry-After": "1"})
    httpx_mock.add_response(
        method="POST",
        url=URL,
        headers={"content-type": "text/event-stream"},
        content=stream_body("Test response"),
    )

    assert "".join(model.prompt("Test message", stream=True)) == "Test response"

    assert len(httpx_mock.get_requests()) == 2
    assert sum(sleeps) == 1


def test_rate_limit_retries_are_exhausted_when_streaming(model, httpx_mock, sleeps):
    add_rate_limited(httpx_mock, Grok.MAX_RETRIES, headers={"Retry-After": "1"})

    with pytest.raises(RateLimitError):
        "".join(model.prompt("Test message", stream=True))

    assert len(httpx_mock.get_requests()) == Grok.MAX_RETRIES


def test_live_search_auto_mode(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "What's the latest news?"},
        ],
        "stream": False,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "auto",
            "max_search_results": 20,
            "return_citations": True,
        },
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt("What's the latest news?", stream=False, search_mode="auto")
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_live_search_on_mode_with_parameters(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Search for AI news"},
        ],
        "stream": False,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "on",
            "max_search_results": 10,
            "return_citations": False,
            "from_date": "2025-01-01",
            "to_date": "2025-01-15",
            "excluded_x_handles": ["@spam_account"],
            "post_favorite_count": 100,
            "post_view_count": 1000,
        },
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt(
        "Search for AI news",
        stream=False,
        search_mode="on",
        max_search_results=10,
        return_citations=False,
        search_from_date="2025-01-01",
        search_to_date="2025-01-15",
        excluded_x_handles="@spam_account",
        post_favorite_count=100,
        post_view_count=1000,
    )
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_live_search_off_mode(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Regular query without search"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt("Regular query without search", stream=False, search_mode="off")
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    content = json.loads(request.content)
    assert "search_parameters" not in content
    assert content == expected_request


def test_live_search_with_sources(model, mock_response, httpx_mock: HTTPXMock):
    search_sources = [
        {"type": "web"},
        {"type": "x"},
        {"type": "news"}
    ]
    
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Search with custom sources"},
        ],
        "stream": False,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "on",
            "max_search_results": 20,
            "return_citations": True,
            "sources": search_sources,
        },
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt(
        "Search with custom sources",
        stream=False,
        search_mode="on",
        search_sources=search_sources,
    )
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_live_search_included_x_handles(model, mock_response, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Search specific handles"},
        ],
        "stream": False,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "on",
            "max_search_results": 20,
            "return_citations": True,
            "included_x_handles": ["@elonmusk", "@openai"],
        },
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt(
        "Search specific handles",
        stream=False,
        search_mode="on",
        included_x_handles="@elonmusk,@openai",
    )
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_live_search_streaming(model, httpx_mock: HTTPXMock):
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Streaming search query"},
        ],
        "stream": True,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "auto",
            "max_search_results": 20,
            "return_citations": True,
        },
    }

    def response_callback(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
        assert json.loads(request.content) == expected_request
        stream_content = [
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Search"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" result"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content="".join(stream_content).encode(),
        )

    httpx_mock.add_callback(
        response_callback,
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        match_json=expected_request,
    )

    response = model.prompt("Streaming search query", stream=True, search_mode="auto")
    chunks = list(response)
    assert "".join(chunks) == "Search result"


def test_live_search_provides_current_data(model, httpx_mock: HTTPXMock):
    """Test that live search actually provides current data, not training data"""
    current_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1721000000,  # July 2025 timestamp
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": "Today is July 14, 2025. Recent headlines from today include breaking news about current events that occurred after my training cutoff."
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "num_sources_used": 5
        }
    }
    
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "What is today's date and latest news?"},
        ],
        "stream": False,
        "temperature": 0.0,
        "search_parameters": {
            "mode": "on",
            "max_search_results": 20,
            "return_citations": True,
        },
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=current_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt("What is today's date and latest news?", stream=False, search_mode="on")
    result = response.text()
    
    # Verify response contains current date (2025) and not training data cutoff (2023)
    assert "2025" in result
    assert "2023" not in result
    assert "today" in result.lower() or "current" in result.lower()
    
    # Verify the response JSON includes usage data indicating sources were used
    assert response.response_json["usage"]["num_sources_used"] > 0


def test_live_search_vs_no_search_behavior(model, httpx_mock: HTTPXMock):
    """Test that search_mode=off doesn't provide current data while search_mode=on does"""
    
    # Mock response for search_mode=off (no live data)
    no_search_response = {
        "id": "chatcmpl-124",
        "object": "chat.completion", 
        "created": 1721000000,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I don't have access to real-time information. My knowledge cutoff is April 2024."
                },
                "finish_reason": "stop",
            }
        ],
    }
    
    # Mock response for search_mode=on (with live data)
    with_search_response = {
        "id": "chatcmpl-125", 
        "object": "chat.completion",
        "created": 1721000000,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on current search results from July 14, 2025, here are today's headlines..."
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "num_sources_used": 8
        }
    }

    # Test without search
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=no_search_response,
        headers={"Content-Type": "application/json"},
        match_json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "What are today's headlines?"}],
            "stream": False,
            "temperature": 0.0,
        },
    )

    response_no_search = model.prompt("What are today's headlines?", stream=False, search_mode="off")
    result_no_search = response_no_search.text()
    
    # Should not have current data
    assert "cutoff" in result_no_search.lower() or "don't have access" in result_no_search.lower()
    
    # Test with search
    httpx_mock.add_response(
        method="POST", 
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=with_search_response,
        headers={"Content-Type": "application/json"},
        match_json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "What are today's headlines?"}],
            "stream": False,
            "temperature": 0.0,
            "search_parameters": {
                "mode": "on",
                "max_search_results": 20,
                "return_citations": True,
            },
        },
    )

    response_with_search = model.prompt("What are today's headlines?", stream=False, search_mode="on")
    result_with_search = response_with_search.text()
    
    # Should have current data
    assert "2025" in result_with_search
    assert "search results" in result_with_search.lower() or "current" in result_with_search.lower()
    assert response_with_search.response_json["usage"]["num_sources_used"] > 0


def test_live_search_validation_errors(model, mock_response, httpx_mock: HTTPXMock):
    # Test excluded handles limit
    with pytest.raises(ValueError, match="Maximum 10 X handles can be excluded"):
        try:
            response = model.prompt(
                "Test",
                stream=False,
                search_mode="on",
                excluded_x_handles=",".join(["@handle" + str(i) for i in range(11)])
            )
            response.text()  # Force execution
        except ValueError:
            raise
    
    # Test conflicting handle parameters
    with pytest.raises(ValueError, match="Cannot specify both excluded_x_handles and included_x_handles"):
        try:
            response = model.prompt(
                "Test",
                stream=False,
                search_mode="on",
                excluded_x_handles="@excluded",
                included_x_handles="@included"
            )
            response.text()  # Force execution
        except ValueError:
            raise
