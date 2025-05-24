"""Main entry point for the FHIR Agent CLI."""

import sys
from .cli.cli import setup_cli

def main():
    """Run the CLI."""
    cli = setup_cli()
    cli()

if __name__ == "__main__":
    sys.exit(main())
