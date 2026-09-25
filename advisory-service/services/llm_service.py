import os
import json

from dotenv import load_dotenv
from groq import Groq


# Load .env
load_dotenv()


# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


# Groq client
client = Groq(api_key=GROQ_API_KEY)


# MODEL_NAME = "openai/gpt-oss-20b"
# MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_NAME = "openai/gpt-oss-120b"


def generate_advisory(
    disease: str,
    confidence: float | None,
    retrieved_documents: list[dict],
):
    """
    Generate a structured advisory using Groq.

    Disease identification is done by the ML service.
    The LLM only explains the detected disease using RAG context.
    """

    # -----------------------------------------
    # Build RAG context
    # -----------------------------------------

    context_parts = []

    for document in retrieved_documents:

        source = document.get(
            "source",
            "Unknown source"
        )

        content = document.get(
            "content",
            ""
        )

        context_parts.append(
            f"""
SOURCE:
{source}

CONTENT:
{content}
"""
        )

    context = "\n---\n".join(context_parts)

    # -----------------------------------------
    # Prompt
    # -----------------------------------------

    prompt = f"""
You are an agricultural advisory assistant
for hydroponic strawberry farming.

The disease has already been identified by a machine-learning
disease detection service.

Detected disease: {disease}
Detection confidence: {confidence}

Use ONLY the supplied knowledge-base context.

IMPORTANT RULES:

1. Do not change the detected disease.
2. Do not diagnose a different disease.
3. Do not invent facts.
4. Do not exaggerate disease severity.
5. Do not provide pesticide/fungicide application rates,
   concentrations, or dosage instructions.
6. Do not claim that a chemical treatment is universally safe.
7. If chemical treatment is mentioned in the knowledge base,
   advise following the applicable local agricultural guidance
   and product label.
8. Prefer practical non-chemical prevention and management.
9. Keep the answer concise and useful for hydroponic growers.

Knowledge-base context:

{context}

Return the advisory using the required JSON structure.
"""

    # -----------------------------------------
    # JSON Schema
    # -----------------------------------------

    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "strawberry_disease_advisory",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string"
                    },
                    "symptoms": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "prevention": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "management": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "hydroponic_considerations": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "summary",
                    "symptoms",
                    "prevention",
                    "management",
                    "hydroponic_considerations"
                ],
                "additionalProperties": False
            }
        }
    }

    # -----------------------------------------
    # Call Groq
    # -----------------------------------------

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # Lower reasoning so more tokens are
        # available for the actual JSON answer.
        reasoning_effort="low",

        temperature=0.2,

        max_completion_tokens=1500,

        response_format=response_schema
    )

    # -----------------------------------------
    # Extract response
    # -----------------------------------------

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned an empty response"
        )

    # Convert JSON string -> Python dictionary
    advisory = json.loads(content)

    return advisory