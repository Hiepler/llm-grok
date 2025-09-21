# llm-grok

[![PyPI](https://img.shields.io/pypi/v/llm-grok.svg)](https://pypi.org/project/llm-grok/)
[![Tests](https://github.com/hiepler/llm-grok/workflows/Test/badge.svg)](https://github.com/hiepler/llm-grok/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hiepler/llm-grok/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) providing access to Grok models using the xAI API

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-grok
```

## Usage

First, obtain an API key from xAI.

Configure the key using the `llm keys set grok` command:

```bash
llm keys set grok
# Paste your xAI API key here
```

You can also set it via environment variable:
```bash
export XAI_API_KEY="your-api-key-here"
```

You can now access the Grok model. Run `llm models` to see it in the list.

To run a prompt through `grok-4-fast` (default model):

```bash
llm -m grok-4-fast 'What is the meaning of life, the universe, and everything?'
```

To start an interactive chat session:

```bash
llm chat -m grok-4-fast
```

Example chat session:
```
Chatting with grok-4-fast
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> Tell me a joke about programming
```

To use a system prompt to give Grok specific instructions:

```bash
cat example.py | llm -m grok-4-fast -s 'explain this code in a humorous way'
```

To set your default model:

```bash
llm models default grok-3-mini-latest
# Now running `llm ...` will use `grok-3-mini-latest` by default
```

## Available Models

The following Grok models are available:

- `grok-4-latest`
- `grok-4-fast` (default)
- `grok-4-fast-reasoning-latest`
- `grok-4-fast-non-reasoning-latest`
- `grok-code-fast-1`
- `grok-3-latest`
- `grok-3-mini-fast-latest`
- `grok-3-mini-latest`
- `grok-3-fast-latest`
- `grok-2-latest`
- `grok-2-vision-latest`

You can check the available models using:
```bash
llm grok models
```

## Model Options

The grok models accept the following options, using `-o name value` syntax:

### Basic Options
* `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher values like 0.8 increase randomness, while lower values like 0.2 make the output more focused and deterministic.
* `-o max_completion_tokens 100`: Maximum number of tokens to generate in the completion (includes both visible tokens and reasoning tokens).

### Live Search Options

All Grok models support live search functionality to access real-time information:

* `-o search_mode auto`: Live search mode. Options: `auto`, `on`, `off` (default: disabled)
* `-o max_search_results 20`: Maximum number of search results to consider (default: 20)
* `-o return_citations true`: Whether to return citations for search results (default: true)
* `-o search_from_date 2025-01-01`: Start date for search results in ISO8601 format (YYYY-MM-DD)
* `-o search_to_date 2025-01-15`: End date for search results in ISO8601 format (YYYY-MM-DD)

### X Platform Search Options
* `-o excluded_x_handles "@spam_account,@another"`: Comma-separated list of X handles to exclude (max 10)
* `-o included_x_handles "@elonmusk,@openai"`: Comma-separated list of X handles to include (cannot be used with excluded_x_handles)
* `-o post_favorite_count 100`: Minimum number of favorites for X posts to be included
* `-o post_view_count 1000`: Minimum number of views for X posts to be included

### Examples

Basic usage with options:
```bash
llm -m grok-4-fast -o temperature 0.2 -o max_completion_tokens 50 'Write a haiku about AI'
```

Using live search to get current information:
```bash
llm -m grok-4-fast -o search_mode on 'What are the latest developments in AI today?'
```

Searching with date constraints:
```bash
llm -m grok-4-fast -o search_mode on -o search_from_date 2025-01-01 -o search_to_date 2025-01-15 'What happened in AI this month?'
```

Filtering X posts by engagement:
```bash
llm -m grok-4-fast -o search_mode on -o post_favorite_count 1000 -o post_view_count 10000 'Show me popular AI discussions on X'
```

Excluding specific X accounts:
```bash
llm -m grok-4-fast -o search_mode on -o excluded_x_handles "@spam_account" 'Latest AI news from X'
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
git clone https://github.com/hiepler/llm-grok.git
cd llm-grok
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:

```bash
pytest
```

## Available Commands

List available Grok models:
```bash
llm grok models
```

## API Documentation

This plugin uses the xAI API. For more information about the API, see:
- [xAI API Documentation](https://docs.x.ai/docs/overview)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0
