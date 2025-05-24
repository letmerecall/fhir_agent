"""FHIR Agent module for processing natural language queries against FHIR servers."""
from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import os
import requests
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

class FHIRQuery(BaseModel):
    """Model representing a parsed FHIR query."""
    resource_type: str = Field(
        default="Patient",
        description="The FHIR resource type to query (e.g., 'Patient', 'Observation')"
    )
    patient_id: Optional[str] = Field(
        None,
        alias="patient",
        description="The patient ID for patient-specific queries"
    )
    code: Optional[str] = Field(
        None,
        description="Medical code (LOINC, SNOMED, etc.) for filtering"
    )
    date: Optional[str] = Field(
        None,
        description="Date or date range for filtering"
    )
    result_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100)"
    )

    class Config:
        # Allow setting fields using field names (not just aliases)
        populate_by_name = True

    def __str__(self) -> str:
        """
        Returns a string representation of the FHIRQuery instance, listing all non-None fields and their values.
        """
        fields = []
        for field_name, _ in self.__fields__.items():
            value = getattr(self, field_name)
            if value is not None:
                fields.append(f"{field_name}={value!r}")
        return f"FHIRQuery({', '.join(fields)})"

    def to_fhir_params(self) -> Dict[str, str]:
        """
        Converts the FHIRQuery instance into a dictionary of FHIR API query parameters.

        Returns:
            A dictionary mapping FHIR parameter names to their string values, suitable for use in FHIR API requests. The `result_count` field is mapped to `_count`, and `patient_id` is mapped to `patient`.
        """
        params = {}
        for field_name, field in self.__fields__.items():
            value = getattr(self, field_name)
            if value is not None and field_name != "resource_type":
                param_name = field.alias if field.alias else field_name
                if param_name == "result_count":
                    params["_count"] = str(value)
                elif param_name == "patient_id":
                    params["patient"] = str(value)
                else:
                    params[param_name] = str(value)
        return params

class FHIRAgent:
    """Agent for processing natural language queries and converting them to FHIR API calls."""

    def __init__(
        self,
        fhir_base_url: str = "http://hapi.fhir.org/baseR4",
        model_name: str = "llama3.2",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initializes the FHIRAgent with configuration for FHIR server access, language model, and prompt parsing.

        Args:
            fhir_base_url: Base URL of the FHIR server to query.
            model_name: Name of the Ollama language model used for parsing natural language queries.
            timeout: Timeout in seconds for HTTP requests and LLM responses.
            headers: Optional HTTP headers to include in FHIR API requests.
        """
        self.fhir_base_url = fhir_base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}

        # Initialize LLM with Ollama
        self.llm = OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
            timeout=timeout
        )

        # Set up the prompt template with format instructions
        self.parser = PydanticOutputParser(pydantic_object=FHIRQuery)

        # Set up the prompt with format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a FHIR query parsing assistant. Your task is to analyze natural language
            queries and extract structured parameters for FHIR API queries.

            Extract the following information from the query:
            - patient_id (REQUIRED): The unique identifier for the patient (e.g., '12345')
            - resource_type: The FHIR resource type (e.g., 'Patient', 'Observation', 'Condition')
            - code: Any medical codes (LOINC, SNOMED) if mentioned
            - date: Date or date range if specified
            - _count: Number of results to return (default: 10)

            Always return a valid JSON object with at least a patient_id and resource_type."""),
            ("human", "Query: {query}\n\nExtract the following parameters in JSON format:\n{format_instructions}")
        ])

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a natural language query and retrieves corresponding FHIR data.

        Args:
            query: A natural language string describing the desired FHIR data (e.g., "Show me patient 12345's latest labs").

        Returns:
            A dictionary containing the status ("success" or "error"), the structured query parameters, the FHIR API response data, and the resource type. On error, includes an error message and the parsed query if available.
        """
        logger.info("\n" + "="*80)
        logger.info("FHIR AGENT - PROCESSING QUERY")
        logger.info("="*80)

        try:
            # Parse the natural language query
            logger.info("[1/3] Parsing natural language query...")
            fhir_query = self._parse_query(query)
            logger.info(f"✅ Successfully parsed query: {fhir_query}")

            # Build the FHIR API URL and parameters
            url = f"{self.fhir_base_url}/{fhir_query.resource_type}"
            params = fhir_query.to_fhir_params()

            logger.info("\n[2/3] Preparing FHIR API request...")
            logger.info(f"  URL: {url}")
            logger.info(f"  Parameters: {params}")

            # Make the FHIR API request
            logger.info("\n[3/3] Making FHIR API request...")
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info("\n✅ FHIR API request successful")
            result = response.json()

            # Print a summary of the results
            if isinstance(result, dict):
                resource_type = result.get("resourceType", "Unknown")
                total = result.get("total", len(result.get("entry", [])))
                logger.info(f"  Found {total} {resource_type} resources")

            return {
                "status": "success",
                "query": fhir_query.dict(),
                "results": result,
                "resource_type": fhir_query.resource_type,
            }

        except Exception as e:
            logger.error("\n❌ Error processing query:")
            logger.error(f"  Type: {type(e).__name__}")
            logger.error(f"  Message: {str(e)}")

            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error("\nResponse content:")
                logger.error(e.response.text)

            return {
                "status": "error",
                "message": f"Failed to process query: {str(e)}",
                "query": fhir_query.dict() if 'fhir_query' in locals() else None
            }

    def _parse_query(self, query: str) -> FHIRQuery:
        """
        Parses a natural language query string into a structured FHIRQuery object.

        Uses a language model and output parser to extract FHIR query parameters from free-text input. Raises a ValueError if parsing fails.

        Args:
            query: The natural language query to be parsed.

        Returns:
            A FHIRQuery object representing the structured query.

        Raises:
            ValueError: If the query cannot be parsed into a valid FHIRQuery.
        """
        logger.info(f"\n=== DEBUG: Parsing query ===")
        logger.info(f"Input query: {query}")

        try:
            # Get the format instructions
            format_instructions = self.parser.get_format_instructions()
            logger.info(f"\nFormat instructions sent to LLM:\n{format_instructions}")

            # Prepare the prompt
            prompt_value = self.prompt.invoke({
                "query": query,
                "format_instructions": format_instructions
            })
            logger.info(f"\nPrompt sent to LLM:\n{prompt_value.to_string()}")

            # Get the LLM response
            response = self.llm.invoke(prompt_value.to_string())
            logger.info(f"\nRaw LLM response:\n{response}")

            # Parse the response
            parsed = self.parser.parse(response)
            logger.info(f"\nParsed query object: {parsed}")

            return parsed

        except Exception as e:
            logger.error(f"\n=== ERROR DETAILS ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise ValueError(f"Failed to parse query: {str(e)}") from e
