import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI,set_trace_processors,OpenAIChatCompletionsModel, RunConfig, function_tool
from langsmith.wrappers import OpenAIAgentsTracingProcessor
# import agentops
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import random
import requests


load_dotenv()



gemini_api_key = os.getenv("GEMINI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")



provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)
# ========== Web Search Agent ==========
web_search_agent = Agent(
    name="WebSearchAgent",
    instructions="You are a tool that performs a web search and returns useful content for the query.",
    model=model,
    # tools=[WebSearchTool()]
)


web_search_agent_as_tool = web_search_agent.as_tool(
    tool_name="WebSearchAgent",
    tool_description="You are a tool that performs a web search and returns useful content for the query.",
)

# ========== Data Analysis Agent ==========
data_analysis_agent = Agent(
    name="DataAnalysisAgent",
    instructions="You are a tool that analyzes data related to the given topic and provides key insights.",
    model=model,
)


data_analysis_agent_as_tool = data_analysis_agent.as_tool(
    tool_name="DataAnalysisAgent",
    tool_description="You are a tool that analyzes climate-related data and provides key insights.",
)

# ========== Writer Agent ==========
writer_agent = Agent(
    name="WriterAgent",
    instructions="You write a formal and detailed report based on the given insights for the user's topic.",
    model=model,
)

writer_agent_as_tool = writer_agent.as_tool(
    tool_name="WriterAgent",
    tool_description="You are a tool that writes a full report based on climate analysis insights. Be formal and detailed."
)

@traceable
async def main():
    main_agent = Agent(
          name="LLM Orchestrator",
          instructions="""
                You are an intelligent orchestrator agent.
                1. Use 'WebSearchAgent' to gather information about the topic the user requested.
                2. Send that information to 'DataAnalysisAgent' to generate insights.
                3. Pass those insights to 'WriterAgent' to generate a final report.
                Be sure to follow the user's topic exactly and not assume it is climate-related.
        """,
         model=model,
         tools=[web_search_agent_as_tool, data_analysis_agent_as_tool, writer_agent_as_tool]
   )
    result = await Runner.run(
            main_agent,
            input="tell me stats of population in karachi. ",
    )

    print(result.final_output)

if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())