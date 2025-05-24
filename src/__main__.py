"""Main entry point for the FHIR Agent CLI."""

import sys
from .cli.cli import setup_cli

def main():
    """
    Initializes and runs the FHIR Agent command-line interface.
    
    Returns:
        The exit status code from the CLI execution.
    """
    cli = setup_cli()
    cli()

if __name__ == "__main__":
    sys.exit(main())
