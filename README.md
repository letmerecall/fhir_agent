# FHIR Agent

Vibe coding a command-line tool for querying FHIR servers using natural language, powered by Ollama and LangChain.

## Features

- Convert natural language queries into FHIR API calls
- Supports common FHIR resources (Patient, Observation, Condition, etc.)
- Uses Ollama for natural language understanding
- Simple and easy-to-use CLI interface
- Fast dependency management with `uv`

## Prerequisites

- Python 3.11 or later
- [Ollama](https://ollama.ai/) (for local LLM inference)
- [uv](https://github.com/astral-sh/uv) (recommended for fast dependency management)

## Quick Start

1. **Clone the repository**:

   ```bash
   git clone git@github.com:letmerecall/fhir_agent.git
   cd fhir_agent
   ```

### Install dependencies using `uv` (Recommended)

```bash
# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with all dependencies
uv pip install -e ".[dev]"
```

## Usage

Run the application using Python's module syntax:

```bash
# Execute a natural language query
python -m src query "Show me the latest lab results for patient 12345"

# Pretty print the JSON output
python -m src query "Find active conditions for patient 67890" --pretty

# Show help
python -m src --help
```

### Configuration

Configure the agent using environment variables in a `.env` file or directly in your shell:

```env
# Required
OLLAMA_BASE_URL=http://localhost:11434  # URL of your Ollama server

# Optional
FHIR_SERVER_URL=http://hapi.fhir.org/baseR4  # Default FHIR server
FHIR_AUTH_TOKEN=your-auth-token            # If authentication is required
OLLAMA_MODEL=llama3                        # Default model to use
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT © 2024 Girish Sharma
