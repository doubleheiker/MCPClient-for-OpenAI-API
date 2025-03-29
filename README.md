# MCPClient-for-OpenAI-API
This is a simple MCP Client **Python implement** for OpenAI API request and response formats.
It can be used to test MCP server with OpenAI API. Any API format follows OpenAI API can use this MCP Client. (eg. NewAPI, OneAPI, etc.)

## Requirements
- mcp package is essential.
- A `.env` file is needed, contains the following:
```
OPENAI_BASE_URL=https://{yourProxyURL}/v1/ # No need for offical openai api key
OPENAI_API_KEY=sk-1234567890abcdefghijlmnopqrstuvwxyz
```

## Functions
**Query**: Type your queries, if need call the mcp server, the LLM will call it.

**Model selection**: It will automanticlly gain the available models from your base url and show in the terminal. You can choose a model to use by yourself. The default model is `gpt-4o`. And you can change the model by inputting `model` to another any time you want.

**Context setting**: The default context size is 5, which means it can remain your (user/tool and assistant) 5 chat as conversation history. You can change the size by inputting `context` and set your own context size.

**Clear context**: When you want to start a new independent query, you can input `clear` to clear the conversation history.

## Usage
```
uv run .\client.py <YourMCPSeverPath>\<servername>.py
```