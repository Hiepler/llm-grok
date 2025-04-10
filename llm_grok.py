import json
from typing import Optional

import click
import httpx
import llm
from pydantic import Field

AVAILABLE_MODELS = [
    "grok-2-latest",
    "grok-2-vision-latest"
]
DEFAULT_MODEL = "grok-2-latest"

@llm.hookimpl
def register_models(register):
    for model_id in AVAILABLE_MODELS:
        register(Grok(model_id))

class Grok(llm.KeyModel):
    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"

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
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []

        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        else:
            messages.append({
                "role": "system",
                "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
            })

        if conversation:
            for prev_response in conversation.responses:
                if prev_response.prompt.system:
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append(
                    {"role": "assistant", "content": prev_response.text()}
                )

        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}

        if not hasattr(prompt, 'options') or not isinstance(prompt.options, self.Options):
            options = self.Options()
        else:
            options = prompt.options

        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "temperature": options.temperature,
        }

        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        try:
            if stream:
                buffer = ""
                with httpx.Client() as client:
                    with client.stream(
                        "POST",
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=None,
                    ) as r:
                        r.raise_for_status()
                        for chunk in r.iter_raw():
                            if chunk:
                                buffer += chunk.decode('utf-8')
                                while '\n\n' in buffer:
                                    message, buffer = buffer.split('\n\n', 1)
                                    if message.startswith('data: '):
                                        data = message[6:]
                                        if data == '[DONE]':
                                            break
                                        try:
                                            parsed = json.loads(data)
                                            if "choices" in parsed and parsed["choices"]:
                                                delta = parsed["choices"][0].get("delta", {})
                                                if "content" in delta:
                                                    content = delta["content"]
                                                    if content:
                                                        yield content
                                        except json.JSONDecodeError:
                                            continue
            else:
                with httpx.Client() as client:
                    r = client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=None,
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    response.response_json = response_data
                    if "choices" in response_data and response_data["choices"]:
                        yield response_data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            error_body = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.json()
                except:
                    error_body = e.response.text
            raise Exception(f"API Error: {str(e)}\nResponse: {error_body}")

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
