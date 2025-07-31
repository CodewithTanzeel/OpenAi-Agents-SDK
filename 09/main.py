import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, AsyncOpenAI,set_trace_processors,OpenAIChatCompletionsModel, RunConfig
from langsmith.wrappers import OpenAIAgentsTracingProcessor
import agentops
from dotenv import load_dotenv
import os

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
# print(LANGSMITH_API_KEY)
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


spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=model,
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=model
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    model = model,
    handoffs=[spanish_agent, english_agent],

)

async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # Expected Output: ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())

