from services.retriever import retrieve_information


disease = "Gray Mold"

query = "What are the symptoms of this disease in strawberries?"


results = retrieve_information(
    disease=disease,
    query=query,
    k=5
)


print("\nDISEASE:")
print(disease)

print("\nQUERY:")
print(query)

print("\nRETRIEVED INFORMATION:\n")


for i, document in enumerate(results, start=1):

    print(f"--- Result {i} ---")

    print("Disease metadata:",
          document.metadata.get("disease"))

    print("Source:",
          document.metadata.get("source"))

    print(document.page_content)

    print()