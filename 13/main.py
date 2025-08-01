# ========== Imports ==========
import os
import asyncio
import random
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_trace_processors,
    OpenAIChatCompletionsModel,
    RunConfig,
    function_tool,
)
from langsmith.wrappers import OpenAIAgentsTracingProcessor
from langsmith import traceable

# ========== Load Environment Variables ==========
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# ========== Configure OpenAI-compatible Gemini Client ==========
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

config = RunConfig(
    model=model,
    model_provider=provider,
)

# ========== Define Agents ==========

web_search_agent = Agent(
    name="WebSearchAgent",
    instructions="You are a tool that performs a web search and returns useful content for the query.",
    model=model,
)

data_analysis_agent = Agent(
    name="DataAnalysisAgent",
    instructions="You are a tool that analyzes data related to the given topic and provides key insights.",
    model=model,
)

writer_agent = Agent(
    name="WriterAgent",
    instructions="You write a formal and detailed report based on the given insights for the user's topic.",
    model=model,
)

# ========== Optional: Convert Agents to Tools ==========

web_search_agent_as_tool = web_search_agent.as_tool(
    tool_name="WebSearchAgent",
    tool_description="You are a tool that performs a web search and returns useful content for the query.",
)

data_analysis_agent_as_tool = data_analysis_agent.as_tool(
    tool_name="DataAnalysisAgent",
    tool_description="You are a tool that analyzes topic-related data and provides key insights.",
)

writer_agent_as_tool = writer_agent.as_tool(
    tool_name="WriterAgent",
    tool_description="You write a detailed report based on the analysis.",
)

# ========== Main Flow with LangSmith Tracing ==========
@traceable
async def run_manual_flow(topic: str):
    # Step 1: Web search
    print("🔍 Step 1: Searching the web...")
    web_search_output = await Runner.run(
        web_search_agent,
        input=f"Search about: {topic}",
        run_config=config
    )
    print("🧾 Search Result:")
    print(web_search_output.final_output)

    # Step 2: Data analysis
    print("\n📊 Step 2: Analyzing the data...")
    data_analysis_output = await Runner.run(
        data_analysis_agent,
        input=f"Analyze this: {web_search_output.final_output}",
        run_config=config
    )
    print("📌 Analysis:")
    print(data_analysis_output.final_output)

    # Step 3: Writing final report
    print("\n🖋️ Step 3: Writing final report...")
    final_report = await Runner.run(
        writer_agent,
        input=f"Write a formal report based on this analysis: {data_analysis_output.final_output}",
        run_config=config
    )
    print("📄 Final Report:")
    print(final_report.final_output)
    return final_report  # returning to access from main


# ========== Entry Point ==========
if __name__ == "__main__":
    topic = input("Enter a topic to generate a detailed report: ")
    set_trace_processors([OpenAIAgentsTracingProcessor()])  # Enable LangSmith tracing
    final_result = asyncio.run(run_manual_flow(topic))
    print("\n✅ Completed Task\n")
    print(final_result.final_output)
