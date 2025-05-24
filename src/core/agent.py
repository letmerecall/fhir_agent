"""FHIR Agent module for processing natural language queries against FHIR servers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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
        """Return a string representation of the query for debugging."""
        fields = []
        for field_name, field in self.__fields__.items():
            value = getattr(self, field_name)
            if value is not None:
                fields.append(f"{field_name}={value!r}")
        return f"FHIRQuery({', '.join(fields)})"

    def to_fhir_params(self) -> Dict[str, str]:
        """Convert the query to FHIR API parameters."""
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
        """Initialize the FHIR agent with a FHIR client and LLM chain.

        Args:
            fhir_base_url: Base URL of the FHIR server
            model_name: Name of the Ollama model to use
            timeout: Request timeout in seconds
            headers: Optional headers for FHIR API requests
        """
        self.fhir_base_url = fhir_base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}

        # Initialize LLM with Ollama
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://localhost:8080",
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

        # Create the chain with proper input variables
        self.chain = ({
            "query": lambda x: x["query"],
            "format_instructions": lambda _: self.parser.get_format_instructions()
        } | self.prompt | self.llm | self.parser)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return FHIR data.

        Args:
            query: Natural language query (e.g., "Show me patient 12345's latest labs")

        Returns:
            Dict containing the query results
        """
        print("\n" + "="*80)
        print("FHIR AGENT - PROCESSING QUERY")
        print("="*80)

        try:
            # Parse the natural language query
            print(f"\n[1/3] Parsing natural language query...")
            fhir_query = self._parse_query(query)
            print(f"✅ Successfully parsed query: {fhir_query}")

            # Build the FHIR API URL and parameters
            url = f"{self.fhir_base_url}/{fhir_query.resource_type}"
            params = fhir_query.to_fhir_params()

            print(f"\n[2/3] Preparing FHIR API request...")
            print(f"  URL: {url}")
            print(f"  Parameters: {params}")

            # Make the FHIR API request
            print("\n[3/3] Making FHIR API request...")
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            print("\n✅ FHIR API request successful")
            result = response.json()

            # Print a summary of the results
            if isinstance(result, dict):
                resource_type = result.get("resourceType", "Unknown")
                total = result.get("total", len(result.get("entry", [])))
                print(f"  Found {total} {resource_type} resources")

            return {
                "status": "success",
                "query": fhir_query.dict(),
                "results": result,
                "resource_type": fhir_query.resource_type,
            }

        except Exception as e:
            print("\n❌ Error processing query:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")

            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print("\nResponse content:")
                print(e.response.text)

            return {
                "status": "error",
                "message": f"Failed to process query: {str(e)}",
                "query": fhir_query.dict() if 'fhir_query' in locals() else None
            }

    def _parse_query(self, query: str) -> FHIRQuery:
        """Parse natural language query into a structured FHIR query.

        Args:
            query: Natural language query string

        Returns:
            FHIRQuery: A structured query object

        Raises:
            ValueError: If the query cannot be parsed
        """
        print(f"\n=== DEBUG: Parsing query ===")
        print(f"Input query: {query}")

        try:
            # Get the format instructions
            format_instructions = self.parser.get_format_instructions()
            print(f"\nFormat instructions sent to LLM:\n{format_instructions}")

            # Prepare the prompt
            prompt_value = self.prompt.invoke({
                "query": query,
                "format_instructions": format_instructions
            })
            print(f"\nPrompt sent to LLM:\n{prompt_value.to_string()}")

            # Get the LLM response
            response = self.llm.invoke(prompt_value.to_string())
            print(f"\nRaw LLM response:\n{response}")

            # Parse the response
            parsed = self.parser.parse(response)
            print(f"\nParsed query object: {parsed}")

            return parsed

        except Exception as e:
            print(f"\n=== ERROR DETAILS ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            raise ValueError(f"Failed to parse query: {str(e)}") from e
