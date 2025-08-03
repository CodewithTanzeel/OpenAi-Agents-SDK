import asyncio
import os
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    set_trace_processors,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from langsmith.wrappers import OpenAIAgentsTracingProcessor
from langsmith import traceable

load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

# === Define Agents for Multi-step Flow ===

code_summarizer = Agent(
    name="code_summarizer",
    instructions="You are a code summarizer. Read the given code and write a short summary explaining what it does in simple English.",
    model=model,
)

translator_agent = Agent(
    name="translator_agent",
    instructions="You are a translator. Translate the given English summary into simple Urdu.",
    model=model,
)

report_generator = Agent(
    name="report_generator",
    instructions="You are a report writer. Format the translated summary into a professional short report.",
    model=model,
)


@traceable
async def process_code_snippet():
    async def run_agent(label, agent, input_data):
        print(f"{label}...")
        result = await Runner.run(agent, input_data)

        # DEBUG: Print what each item contains
        for i, item in enumerate(result.new_items):
            print(f"🔍 Raw item {i+1} type: {type(item)}, content: {item}")

        # Join raw str(item) if text/content attributes don't exist
        output = "\n".join(str(item) for item in result.new_items)
        print(output)
        return output

    code = "def greet(name): return f'Hello, {name}!'"

    summary_text = await run_agent("⏳ Step 1: Summarizing code", code_summarizer, code)
    translated_text = await run_agent(
        "🌐 Step 2: Translating summary to Urdu", translator_agent, summary_text
    )
    await run_agent(
        "🗂️ Step 3: Generating final report", report_generator, translated_text
    )
    


if __name__ == "__main__":
    asyncio.run(process_code_snippet())
    set_trace_processors([OpenAIAgentsTracingProcessor()])
