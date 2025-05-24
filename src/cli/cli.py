"""Command-line interface for the FHIR Agent."""

import json
import os
import sys
from typing import Any, Dict, Optional

import click

from ..core.agent import FHIRAgent

def format_result(result: Dict[str, Any], pretty: bool = False) -> str:
    """Format the result for CLI output."""
    if result.get("status") == "error":
        return click.style(f"Error: {result.get('message', 'Unknown error')}", fg="red")

    if pretty:
        return json.dumps(result, indent=2, default=str)
    return json.dumps(result, default=str)

def setup_cli() -> click.Group:
    """Set up the command line interface."""
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
        """Execute a natural language FHIR query"""
        try:
            result = agent.process_query(query)
            click.echo(format_result(result, pretty))
        except Exception as e:
            click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)
            sys.exit(1)

    return cli
