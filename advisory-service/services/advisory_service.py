from services.retriever import retrieve_information
from services.llm_service import generate_advisory


def create_advisory(
    disease: str,
    confidence: float | None = None
):

    query = f"""
    Provide information about {disease} in hydroponic strawberries,
    including description, symptoms, causes, prevention,
    management and hydroponic considerations.
    """

    documents = retrieve_information(
        disease=disease,
        query=query,
        k=5
    )

    retrieved_information = []

    for document in documents:
        retrieved_information.append({
            "source": document.metadata.get("source"),
            "content": document.page_content
        })

    advisory = generate_advisory(
        disease=disease,
        confidence=confidence,
        retrieved_documents=retrieved_information
    )

    return advisory