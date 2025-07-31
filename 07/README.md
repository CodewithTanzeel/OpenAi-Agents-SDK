# Gemini Agent with OpenAI Agents SDK

This project demonstrates how to use Google's Gemini API with the OpenAI Agents SDK.

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up your Gemini API key:**
   ```bash
   uv run python setup_env.py
   ```
   
   Then edit the `.env` file and replace `your_gemini_api_key_here` with your actual Gemini API key.

3. **Get your Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key and paste it in your `.env` file

## Run the Agent

```bash
uv run python main.py
```

## What it does

The agent will:
- Use the Gemini 2.5 Flash model via the OpenAI-compatible interface
- Tell you 5 jokes when prompted
- Stream the response in real-time

## Troubleshooting

- **"GEMINI_API_KEY not found"**: Make sure you've set up the `.env` file with your API key
- **API errors**: Verify your Gemini API key is correct and has sufficient quota
- **Import errors**: Run `uv sync` to install all dependencies

## Files

- `main.py`: Main agent code using Gemini API
- `setup_env.py`: Helper script to create the `.env` file
- `pyproject.toml`: Project dependencies
- `.env`: Your API key (create this with the setup script)
