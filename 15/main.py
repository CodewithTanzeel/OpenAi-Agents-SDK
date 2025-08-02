import asyncio
import os
from dotenv import load_dotenv

from agents import (
    Agent,
    ItemHelpers,
    Runner,
    trace,
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
async def process_code(code_snippet: str):
    print("⏳ Step 1: Summarizing code...")
    summary = await Runner.run(code_summarizer, code_snippet)
    summary_text = summary.final_output
    print(f"\n📝 Code Summary:\n{summary_text}")

    print("\n🌐 Step 2: Translating summary to Urdu...")
    translated = await Runner.run(translator_agent, summary_text)
    translated_text = translated.final_output
    print(f"\n🌍 Urdu Translation:\n{translated_text}")

    print("\n🗂️ Step 3: Generating final report...")
    report = await Runner.run(report_generator, translated_text)
    final_report = report.final_output
    print(f"\n✅ Final Report:\n{final_report}")


# === Run ===
if __name__ == "__main__":
    example_code = """
def is_even(n):
    return n % 2 == 0
"""
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(process_code(example_code))
