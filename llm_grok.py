import json
import sys
import time
from typing import Optional

import click
import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

AVAILABLE_MODELS = [
    "grok-code-fast-1",
    "grok-4-latest",
    "grok-4-1-fast",
    "grok-4-1-fast-reasoning-latest",
    "grok-4-1-fast-non-reasoning-latest",
    "grok-4-fast",
    "grok-4-fast-reasoning-latest",
    "grok-4-fast-non-reasoning-latest",
    "grok-3-latest",
    "grok-3-fast-latest",
    "grok-3-mini-latest",
    "grok-3-mini-fast-latest",
    "grok-2-latest",
    "grok-2-vision-latest",
]
DEFAULT_MODEL = "grok-4-1-fast"


@llm.hookimpl
def register_models(register):
    for model_id in AVAILABLE_MODELS:
        register(Grok(model_id))


class GrokError(Exception):
    """Base exception for Grok API errors"""

    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(message)


class RateLimitError(GrokError):
    """Exception for rate limit errors"""

    pass


class QuotaExceededError(GrokError):
    """Exception for quota exceeded errors"""

    pass


class Grok(llm.KeyModel):
    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"
    supports_tools = True
    MAX_RETRIES = 3
    BASE_DELAY = 1  # Base delay in seconds

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.0,
        )
        max_completion_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate, including visible output tokens and reasoning tokens.",
            ge=0,
            default=None,
        )
        search_mode: Optional[str] = Field(
            description="Live search mode: 'auto', 'on', or 'off' (default: off)",
            default=None,
        )
        max_search_results: Optional[int] = Field(
            description="Maximum number of search results to consider (default: 20)",
            ge=1,
            default=20,
        )
        search_from_date: Optional[str] = Field(
            description="Start date for search results in ISO8601 format (YYYY-MM-DD)",
            default=None,
        )
        search_to_date: Optional[str] = Field(
            description="End date for search results in ISO8601 format (YYYY-MM-DD)",
            default=None,
        )
        return_citations: Optional[bool] = Field(
            description="Whether to return citations for search results (default: True)",
            default=True,
        )
        search_sources: Optional[list] = Field(
            description="List of search sources to use (default: web and x)",
            default=None,
        )
        excluded_x_handles: Optional[str] = Field(
            description="Comma-separated list of X handles to exclude from search (max 10)",
            default=None,
        )
        included_x_handles: Optional[str] = Field(
            description="Comma-separated list of X handles to include in search (cannot be used with excluded_x_handles)",
            default=None,
        )
        post_favorite_count: Optional[int] = Field(
            description="Minimum number of favorites for X posts to be included",
            ge=0,
            default=None,
        )
        post_view_count: Optional[int] = Field(
            description="Minimum number of views for X posts to be included",
            ge=0,
            default=None,
        )
        # TODO: Add reasoning_effort

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []

        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        if conversation:
            for prev_response in conversation.responses:
                if prev_response.prompt.system:
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                # Add tool_results before user message (they belong to the previous turn)
                if prev_response.prompt.tool_results:
                    for tool_result in prev_response.prompt.tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": tool_result.output,
                        })
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )

                # Add assistant response with tool_calls if present
                tool_calls = prev_response.tool_calls()
                content = prev_response.text()
                assistant_message = {"role": "assistant"}
                if content is not None:
                    assistant_message["content"] = content
                if tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_message)

        # Add tool_results before final user message (they belong to the previous turn)
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "content": tool_result.output,
                })

        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def _handle_rate_limit(self, response, attempt):
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                wait_time = int(retry_after)
                if attempt < self.MAX_RETRIES - 1:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Rate limit hit. Waiting {wait_time}s as suggested by API...",
                            total=wait_time,
                        )
                        while not progress.finished:
                            time.sleep(1)
                            progress.update(task, advance=1)
                    return True
            except ValueError:
                pass

        if attempt < self.MAX_RETRIES - 1:
            delay = self.BASE_DELAY * (2**attempt)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Rate limit hit. Retrying in {delay}s...", total=delay
                )
                while not progress.finished:
                    time.sleep(1)
                    progress.update(task, advance=1)
            return True

        try:
            error_details = response.json()
            if "error" in error_details:
                error_message = error_details["error"].get("message", "")
                if (
                    "quota exceeded" in error_message.lower()
                    or "insufficient credits" in error_message.lower()
                ):
                    raise QuotaExceededError(
                        "API Quota Exceeded",
                        "Your x.ai API quota has been exceeded or you have insufficient credits.\n"
                        "Please visit https://x.ai to check your account status.",
                    )
        except:
            pass

        raise RateLimitError(
            "Rate Limit Exceeded",
            "You've hit the API rate limit. This could mean:\n"
            "1. Too many requests in a short time\n"
            "2. Your account has run out of credits\n\n"
            "Please visit https://x.ai to check your account status\n"
            "or wait a few minutes before trying again.",
        )

    def _make_request(self, client, method, url, headers, json_data, stream=False):
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return client.stream(
                        method, url, headers=headers, json=json_data, timeout=None
                    )
                else:
                    return client.request(
                        method, url, headers=headers, json=json_data, timeout=None
                    )
            except httpx.HTTPError as e:
                if (
                    hasattr(e, "response")
                    and e.response is not None
                    and e.response.status_code == 429
                ):
                    if self._handle_rate_limit(e.response, attempt):
                        continue
                raise

    def execute(self, prompt, stream, response, conversation, key=None):
        key = self.get_key(key)
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}

        if not hasattr(prompt, "options") or not isinstance(
            prompt.options, self.Options
        ):
            options = self.Options()
        else:
            options = prompt.options

        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "temperature": options.temperature,
        }

        if options.max_completion_tokens is not None:
            # TODO: If max_completion_tokens runs out during reasoning, llm will crash when trying to log to db
            body["max_completion_tokens"] = options.max_completion_tokens

        # Add live search parameters if search is enabled
        if options.search_mode and options.search_mode != "off":
            search_params = {
                "mode": options.search_mode if options.search_mode else "auto",
                "max_search_results": options.max_search_results,
                "return_citations": options.return_citations,
            }
            
            # Add date range if specified
            if options.search_from_date:
                search_params["from_date"] = options.search_from_date
            if options.search_to_date:
                search_params["to_date"] = options.search_to_date
            
            # Add sources if specified
            if options.search_sources:
                search_params["sources"] = options.search_sources
            
            # Validate conflicting handle parameters first
            if options.excluded_x_handles and options.included_x_handles:
                raise ValueError("Cannot specify both excluded_x_handles and included_x_handles")

            # Add X-specific parameters
            if options.excluded_x_handles:
                excluded_handles = options.excluded_x_handles
                if isinstance(excluded_handles, str):
                    excluded_handles = [h.strip() for h in excluded_handles.split(",")]
                if len(excluded_handles) > 10:
                    raise ValueError("Maximum 10 X handles can be excluded")
                search_params["excluded_x_handles"] = excluded_handles
            elif options.included_x_handles:
                included_handles = options.included_x_handles
                if isinstance(included_handles, str):
                    included_handles = [h.strip() for h in included_handles.split(",")]
                search_params["included_x_handles"] = included_handles
            
            if options.post_favorite_count is not None:
                search_params["post_favorite_count"] = options.post_favorite_count
            
            if options.post_view_count is not None:
                search_params["post_view_count"] = options.post_view_count
            
            body["search_parameters"] = search_params

        # Add tools to request body if present
        if prompt.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in prompt.tools
            ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        try:
            if stream:
                buffer = ""
                tool_calls_by_index = {}
                with httpx.Client() as client:
                    with self._make_request(
                        client,
                        "POST",
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json_data=body,
                        stream=True,
                    ) as r:
                        r.raise_for_status()
                        for chunk in r.iter_raw():
                            if chunk:
                                buffer += chunk.decode("utf-8")
                                while "\n\n" in buffer:
                                    message, buffer = buffer.split("\n\n", 1)
                                    if message.startswith("data: "):
                                        data = message[6:]
                                        if data == "[DONE]":
                                            break
                                        try:
                                            parsed = json.loads(data)
                                            if (
                                                "choices" in parsed
                                                and parsed["choices"]
                                            ):
                                                delta = parsed["choices"][0].get(
                                                    "delta", {}
                                                )
                                                if "content" in delta:
                                                    content = delta["content"]
                                                    if content:
                                                        yield content
                                                # Handle tool_calls in streaming response
                                                if "tool_calls" in delta:
                                                    for tool_call_delta in delta["tool_calls"]:
                                                        index = tool_call_delta.get("index")
                                                        if index is not None:
                                                            if index not in tool_calls_by_index:
                                                                tool_calls_by_index[index] = {
                                                                    "id": "",
                                                                    "type": "function",
                                                                    "function": {"name": "", "arguments": ""},
                                                                }
                                                            if "id" in tool_call_delta:
                                                                tool_calls_by_index[index]["id"] = tool_call_delta["id"]
                                                            if "function" in tool_call_delta:
                                                                func_delta = tool_call_delta["function"]
                                                                if "name" in func_delta:
                                                                    tool_calls_by_index[index]["function"]["name"] = func_delta["name"]
                                                                if "arguments" in func_delta:
                                                                    tool_calls_by_index[index]["function"]["arguments"] += func_delta["arguments"]
                                        except json.JSONDecodeError:
                                            continue
                # Add accumulated tool_calls to response
                for index in sorted(tool_calls_by_index.keys()):
                    tc = tool_calls_by_index[index]
                    if tc["id"] and tc["function"]["name"]:
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=tc["id"],
                                name=tc["function"]["name"],
                                arguments=json.loads(tc["function"]["arguments"]),
                            )
                        )
            else:
                with httpx.Client() as client:
                    r = self._make_request(
                        client,
                        "POST",
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json_data=body,
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    response.response_json = response_data
                    if "choices" in response_data and response_data["choices"]:
                        message = response_data["choices"][0]["message"]
                        # Parse tool_calls from non-streaming response
                        if "tool_calls" in message:
                            for tool_call_data in message["tool_calls"]:
                                response.add_tool_call(
                                    llm.ToolCall(
                                        tool_call_id=tool_call_data["id"],
                                        name=tool_call_data["function"]["name"],
                                        arguments=json.loads(tool_call_data["function"]["arguments"]),
                                    )
                                )
                        if "content" in message:
                            yield message["content"]
        except httpx.HTTPError as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 429
            ):
                try:
                    self._handle_rate_limit(e.response, self.MAX_RETRIES)
                except (RateLimitError, QuotaExceededError) as rate_error:
                    error_panel = Panel.fit(
                        f"[bold red]{rate_error.message}[/]\n\n[white]{rate_error.details}[/]",
                        title="❌ Error",
                        border_style="red",
                    )
                    if "pytest" in sys.modules:
                        raise rate_error
                    rprint(error_panel)
                    sys.exit(1)

            error_body = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    if e.response.is_stream_consumed:
                        error_body = e.response.text
                    else:
                        error_body = e.response.read().decode("utf-8")
                except:
                    error_body = str(e)

            error_message = f"API Error: {str(e)}"
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if "error" in error_json and "message" in error_json["error"]:
                        error_message = error_json["error"]["message"]
                except:
                    pass

            error_panel = Panel.fit(
                f"[bold red]API Error[/]\n\n[white]{error_message}[/]",
                title="❌ Error",
                border_style="red",
            )
            if "pytest" in sys.modules:
                raise GrokError(error_message)
            rprint(error_panel)
            sys.exit(1)


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def grok():
        "Commands for the Grok model"

    @grok.command()
    def models():
        "Show available Grok models"
        click.echo("Available models:")
        for model in AVAILABLE_MODELS:
            if model == DEFAULT_MODEL:
                click.echo(f"  {model} (default)")
            else:
                click.echo(f"  {model}")
