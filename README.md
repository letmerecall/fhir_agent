# FHIR Agent

Vibe coding a command-line tool for querying FHIR servers using natural language.

## Features

- Convert natural language queries into FHIR API calls
- Supports common FHIR resources (Patient, Observation, Condition, etc.)
- Uses Ollama for natural language understanding
- Simple and easy-to-use CLI interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fhir-agent.git
   cd fhir-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```bash
# Execute a natural language query
fhir-agent query "Show me the latest lab results for patient 12345"

# Pretty print the JSON output
fhir-agent query "Find active conditions for patient 67890" --pretty
```

### Environment Variables

You can configure the agent using the following environment variables:

- `FHIR_SERVER_URL`: Base URL of the FHIR server (default: `http://hapi.fhir.org/baseR4`)
- `FHIR_AUTH_TOKEN`: Optional authentication token for the FHIR server
- `OLLAMA_MODEL`: Name of the Ollama model to use (default: `llama3`)

## Development

### Dependencies

- Python 3.11+
- [Ollama](https://ollama.ai/) (for local LLM inference)

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

## Credits

This project was developed with the assistance of:

- [Windsurf](https://windsurf.com) - AI development platform
- [Cursor](https://www.cursor.com) - AI-powered code editor
- [Warp](https://www.warp.dev) - Modern terminal with AI capabilities

## License

MIT
