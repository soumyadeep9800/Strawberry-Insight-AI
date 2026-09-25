from services.retriever import retrieve_information
from services.llm_service import generate_advisory


def main():

    # -----------------------------------------
    # Disease detection result
    # -----------------------------------------

    disease = "Gray Mold"
    confidence = 0.91

    print("======================================")
    print("TESTING ADVISORY LLM")
    print("======================================")

    print(f"\nDisease: {disease}")
    print(f"Confidence: {confidence}")

    # -----------------------------------------
    # Retrieve information from ChromaDB
    # -----------------------------------------

    query = f"""
    Provide information about {disease} in hydroponic strawberries,
    including description, symptoms, causes, prevention,
    management and hydroponic considerations.
    """

    print("\nRetrieving information from ChromaDB...")

    documents = retrieve_information(
        disease=disease,
        query=query,
        k=5
    )

    print(
        f"Retrieved {len(documents)} documents."
    )

    # -----------------------------------------
    # Convert LangChain documents
    # -----------------------------------------

    retrieved_information = []

    for document in documents:

        retrieved_information.append({
            "source": document.metadata.get("source"),
            "content": document.page_content
        })

    # -----------------------------------------
    # Generate advisory using Groq
    # -----------------------------------------

    print("\nSending retrieved information to Groq...")

    advisory = generate_advisory(
        disease=disease,
        confidence=confidence,
        retrieved_documents=retrieved_information
    )

    # -----------------------------------------
    # Display result
    # -----------------------------------------

    print("\n======================================")
    print("LLM ADVISORY")
    print("======================================")

    print("\nSummary:")
    print(advisory["summary"])

    print("\nSymptoms:")

    for symptom in advisory["symptoms"]:
        print(f"- {symptom}")

    print("\nPrevention:")

    for item in advisory["prevention"]:
        print(f"- {item}")

    print("\nManagement:")

    for item in advisory["management"]:
        print(f"- {item}")

    print("\nHydroponic Considerations:")

    for item in advisory["hydroponic_considerations"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()