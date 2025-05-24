"""Command-line interface for the FHIR Agent."""

import json
import os
import sys
from typing import Any, Dict, Optional

import click

from ..core.agent import FHIRAgent

def format_result(result: Dict[str, Any], pretty: bool = False) -> str:
    """
    Formats the result dictionary for CLI output, displaying errors in red or serializing as JSON.
    
    Args:
        result: The result dictionary to format, expected to contain a "status" key.
        pretty: If True, outputs indented (pretty-printed) JSON.
    
    Returns:
        A string suitable for CLI display, with errors highlighted in red or successful results as JSON.
    """
    if result.get("status") == "error":
        return click.style(f"Error: {result.get('message', 'Unknown error')}", fg="red")

    if pretty:
        return json.dumps(result, indent=2, default=str)
    return json.dumps(result, default=str)

def setup_cli() -> click.Group:
    """
    Initializes and configures the command-line interface for querying FHIR servers using natural language.
    
    Returns:
        A Click command group with commands for interacting with the FHIR Agent.
    """
    agent = FHIRAgent(
        fhir_base_url=os.getenv("FHIR_SERVER_URL", "http://hapi.fhir.org/baseR4"),
        model_name=os.getenv("OLLAMA_MODEL", "llama3.2"),
        headers={
            "Authorization": f"Bearer {os.getenv('FHIR_AUTH_TOKEN', '')}"
            if os.getenv("FHIR_AUTH_TOKEN")
            else None
        }
    )

    @click.group()
    def cli():
        """FHIR Agent - Query FHIR servers using natural language."""
        pass

    @cli.command()
    @click.argument("query")
    @click.option("--pretty", is_flag=True, help="Pretty print JSON output")
    def query(query: str, pretty: bool):
        """
        Executes a natural language query against the FHIR Agent and displays the result.
        
        Args:
            query: The natural language query to be processed.
            pretty: If True, formats the output as pretty-printed JSON.
        
        On error, prints the error message in red and exits with a non-zero status code.
        """
        try:
            result = agent.process_query(query)
            click.echo(format_result(result, pretty))
        except Exception as e:
            click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)
            sys.exit(1)

    return cli
