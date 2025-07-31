import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please set your Gemini API key in the .env file:")
    print("GEMINI_API_KEY=your_actual_api_key_here")
    exit(1)

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini API endpoint
)

# Define the chat model using the Gemini model via the OpenAI-compatible interface
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client)
    

config = RunConfig(
    model=model,
    tracing_disabled=True,  # Disable tracing for this run
)

async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=model,  # Explicitly set the model for the agent
    )

    result = Runner.run_streamed(agent, input="Write 1200 words about the history of the Islam.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())