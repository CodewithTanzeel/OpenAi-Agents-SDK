from agents import Runner,Agent, OpenAIChatCompletionsModel,AsyncOpenAI,RunConfig
import os 
from dotenv import load_dotenv
import asyncio



load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")



external_client:AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model:OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)
agent = Agent(
    name = "ExpertProgrammer",
    instructions="Give Expert programming advices and Code Suggestions",
    model=model
)

result = Runner.run_sync(agent,"tell me about aggreagation compostion and association")

print(result.final_output)


