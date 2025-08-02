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
async def process_code_snippet(code_snippet: str):
    print("⏳ Step 1: Summarizing code...")
    # Step 1: Summarize
    summary_result = await Runner.run(code_summarizer, code_snippet)
    summary_text = ItemHelpers.text_message_outputs(summary_result.new_items)
    print("📝 Code Summary:\n", summary_text)

    # Step 2: Translate
    print("🌐 Step 2: Translating summary to Urdu...")
    translation_result = await Runner.run(translator_agent, summary_text)
    translated_text = ItemHelpers.text_message_outputs(translation_result.new_items)
    print("🌍 Urdu Translation:\n", translated_text)

    # Step 3: Final Report
    print("🗂️ Step 3: Generating final report...")
    report_result = await Runner.run(report_generator, translated_text)
    final_report = ItemHelpers.text_message_outputs(report_result.new_items)

    print("✅ Final Report:\n", final_report)


# Example call
if __name__ == "__main__":
    sample_code = "def greet(name): return f'Hello, {name}!'"
    asyncio.run(process_code_snippet(sample_code))
    set_trace_processors([OpenAIAgentsTracingProcessor()])
