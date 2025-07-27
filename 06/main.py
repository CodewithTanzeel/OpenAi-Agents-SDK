# Import necessary classes and functions from the agents package
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
# Import standard and third-party libraries
import os 
from dotenv import load_dotenv
import asyncio
import chainlit as cl

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Gemini API key from environment variables
# Make sure you have GEMINI_API_KEY set in your .env file
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the external OpenAI-compatible client for Gemini
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini API endpoint
)

# Define the chat model using the Gemini model via the OpenAI-compatible interface
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client)

# Set up the run configuration for the agent
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,  # Disable tracing for this run
)

# Create an agent with a name and instructions for its behavior
agent = Agent(
    name = "ExpertProgrammer",
    instructions="Give Expert programming advices and Code Suggestions",
    model=model
)

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("History", [])
    await cl.Message(content="Hello, I'm ExpertProgrammer, I'm here to help you with your programming questions.").send()


# Define the message handler for Chainlit
@cl.on_message
async def handle_message(message: cl.Message):
    # Run the agent with the user's input and configuration
    history = cl.user_session.get("History")
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("History", history)
    
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("History", history)
    # Send the agent's final output as a response message
    await cl.Message(content=result.final_output).send()

