import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,function_tool
import os
from dotenv import load_dotenv
import requests
import random
import asyncio

load_dotenv()



gemini_api_key = os.getenv("GEMINI_API_KEY")
LANGSMITH_API_KEYd = os.getenv("LANGCHAIN_API_KEY")



provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider
)

@function_tool
def how_many_jokes():
    """
    Get Random Number for jokes
    """
    return random.randint(1, 10)

@function_tool
def get_weather(city: str) -> str:
    """
    Get the weather for a given city
    """
    try:
        result = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}"
        )

        data = result.json()

        return f"The current weather in {city} is {data["current"]["temp_c"]}C with {data["current"]["condition"]["text"]}."
    
    except Exception as e :
        return f"Could not fetch weather data due to {e}"


@traceable
async def main():
    agent = Agent(
        name="Assistant",
        instructions="""
                if the user asks for jokes, first call 'how_many_jokes' function, then tell that jokess with numbers.
                if the user asks for weather, call the 'get_weather' funciton with city name
            """,
            model=model,
            tools=[get_weather, how_many_jokes]
    )
    result = await Runner.run(
        agent,
        input="Please provide a weather report for Karachi, Pakistan. ",
    )

    print(result.final_output)

if __name__ == "__main__":

    asyncio.run(main())