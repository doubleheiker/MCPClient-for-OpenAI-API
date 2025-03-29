import asyncio
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.available_models: List[str] = []
        self.selected_model: str = "gpt-4o"  # Default model
        self.context_size: int = 5  # Default context size
        self.conversation_history: List[Dict[str, str]] = []  # Store conversation history

    async def get_available_models(self) -> List[str]:
        """Fetch available models from the OpenAI base URL"""
        try:
            models_list = self.openai.models.list()
            self.available_models = [model.id for model in models_list.data]
            return self.available_models
        except Exception as e:
            print(f"Error fetching models: {str(e)}")
            return []
        
    async def select_model(self) -> str:
        """Display available models and let user select one"""
        models = await self.get_available_models()
        
        if not models:
            print("No models available. Please check your base url! Using default model.")
            return self.selected_model
            
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
            
        while True:
            try:
                choice = input("\nSelect model (number or name), or press Enter for default model: ").strip()
                
                # Use default if empty
                if not choice:
                    print(f"Using default model: {self.selected_model}")
                    return self.selected_model
                
                # Try to parse as number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        self.selected_model = models[idx]
                        print(f"Selected model: {self.selected_model}")
                        return self.selected_model
                    else:
                        print(f"Invalid number. Please enter 1-{len(models)}.")
                # Try direct model name
                elif choice in models:
                    self.selected_model = choice
                    print(f"Selected model: {self.selected_model}")
                    return self.selected_model
                else:
                    print("Invalid model selection. Please try again.")
            except Exception as e:
                print(f"Error selecting model: {str(e)}")
    
    async def set_context_size(self) -> None:
        """Let user set the conversation context size"""
        while True:
            try:
                size = input(f"\nEnter context size (current: {self.context_size}): ").strip()
                
                # Keep current if empty
                if not size:
                    print(f"Keeping current context size: {self.context_size}")
                    return
                
                size = int(size)
                if size < 1:
                    print("Context size must be at least 1.")
                    continue
                    
                self.context_size = size
                # Trim existing history if needed
                if len(self.conversation_history) > self.context_size * 2:
                    self.conversation_history = self.conversation_history[-(self.context_size * 2):]
                    
                print(f"Context size set to: {self.context_size}")
                return
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"Error setting context size: {str(e)}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""

        # Add the new query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Get the relevant conversation history based on context size
        # Each conversation pair has a user/tool message and an assistant response
        context_message_count = min(len(self.conversation_history), self.context_size * 2)
        messages = self.conversation_history[-context_message_count:]

        ###------ This is for no history ------###
        # messages = [
        #     {
        #         "role": "user",
        #         "content": query
        #     }
        # ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "type":"function",
            "function":{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initial Openai API call
        response = self.openai.chat.completions.create(
            model=self.selected_model,
            max_completion_tokens=1000,
            messages=messages,
            tools=available_tools,
            tool_choice="auto"  # Automatically choose the best tool
        )
        # print(f"response: {response}\n")

        # Process response and handle tool calls
        final_text = []

        for choice in response.choices:
            message = choice.message
            is_function_call = message.tool_calls
            if not is_function_call:
                final_text.append(message.content)
                # Add assistant's response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
            else:
                # Add the assistant's tool call request to conversation history
                self.conversation_history.append(message)
                
                # Extract tool call details
                tool_id = message.tool_calls[0].id
                tool_name = message.tool_calls[0].function.name
                tool_args = json.loads(message.tool_calls[0].function.arguments)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"Tool call result: {result.content}\n")
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Add the tool response to conversation history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result.content
                })

                # Update messages with the latest conversation
                messages = self.conversation_history[-self.context_size * 2:]

                ###------ This is for no history ------###
                # messages.append(message)
                # messages.append({
                #   "role": "tool",
                #   "tool_call_id": tool_id,
                #   "name": tool_name,
                #   "content":  result.content
                # })
                # print(f"messages: {messages}\n")

                # Get next response from LLMs
                response = self.openai.chat.completions.create(
                    model=self.selected_model,
                    max_completion_tokens=1000,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto"
                )
                # print(f"response: {response}\n")

                final_response_text = response.choices[0].message.content
                final_text.append(final_response_text)

                # Add the final assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response_text
                })

        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'model' to change model, 'context' to set context size, 'clear' to clear conversation history, or 'quit' to exit.")

        # Select model at startup
        await self.select_model()

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'model':
                    await self.select_model()
                    continue
                elif query.lower() == 'context':
                    await self.set_context_size()
                    continue
                elif query.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.")
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())