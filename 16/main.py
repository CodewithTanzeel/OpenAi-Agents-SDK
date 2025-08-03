import asyncio
import os
from dotenv import load_dotenv

from agents import (
    Agent,
    ItemHelpers,
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
    print("⏳ Step 1: Summarizing code...")
    summary_result = await Runner.run(
        code_summarizer, "def greet(name): return f'Hello, {name}!'"
    )
    summary_text = "\n".join(item.content for item in summary_result.new_items)
    print("📝 Code Summary:\n", summary_text)

    print("🌐 Step 2: Translating summary to Urdu...")
    translation_result = await Runner.run(translator_agent, summary_text)
    translated_text = "\n".join(item.content for item in translation_result.new_items)
    print("🌍 Urdu Translation:\n", translated_text)

    print("🗂️ Step 3: Generating final report...")
    report_result = await Runner.run(report_generator, translated_text)
    final_report = "\n".join(item.content for item in report_result.new_items)
    print("✅ Final Report:\n", final_report)


if __name__ == "__main__":
    asyncio.run(process_code_snippet())
    set_trace_processors([OpenAIAgentsTracingProcessor()])
