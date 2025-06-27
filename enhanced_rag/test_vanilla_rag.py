import os
from dotenv import load_dotenv
from src.vanilla_rag import VanillaRAG

load_dotenv()

# Test configuration
pdf_path = r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\pdfs_train\eis_Continental_United_States_Interceptor_Site.pdf"
model_name = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")
parent_model = "openai"

# Initialize VanillaRAG
print("Initializing VanillaRAG...")
vanilla_rag = VanillaRAG(pdf_path, model_name, api_key, parent_model)

# Get chunk information
print("\n=== CHUNK INFORMATION ===")
chunk_info = vanilla_rag.get_chunks_info()
print(f"Total chunks: {chunk_info['total_chunks']}")
print("\nFirst few chunks:")
for i, chunk in enumerate(chunk_info['first_few_chunks']):
    print(f"Chunk {i+1} - Page {chunk['page']}: {chunk['content_preview']}")

# Test queries
test_queries = [
    "Did the EIS process for the CRJMTC site include participation from the Ohio Department of Education?",
    "Was the Maine Historic Preservation Commission present during the EIS process for the SERE East Site on May 15, 2014?",
    "Is the median household income below the state average?"
]

print("\n=== TESTING RETRIEVAL ===")
for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: {query}")
    print("-" * 50)
    
    # Get retrieved documents directly
    retrieved_docs = vanilla_rag.retriever.invoke(query)
    
    print(f"Retrieved {len(retrieved_docs)} chunks:")
    for j, doc in enumerate(retrieved_docs):
        page = doc.metadata.get("page", "unknown")
        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        print(f"  Chunk {j+1} (Page {page}):\n {content_preview}")
        print()
    
    # Check if chunks are identical
    if len(retrieved_docs) > 1:
        identical_chunks = []
        for j in range(len(retrieved_docs)):
            for k in range(j+1, len(retrieved_docs)):
                if retrieved_docs[j].page_content == retrieved_docs[k].page_content:
                    identical_chunks.append((j+1, k+1))
        
        if identical_chunks:
            print(f"⚠️  WARNING: Found identical chunks: {identical_chunks}")
        else:
            print("✅ All retrieved chunks are different")
    
    print("=" * 80)

print("\n=== TESTING FULL QUERY ===")
response = vanilla_rag.query(test_queries[0])
print(f"Answer: {response['answer']}")
print(f"Retrieved {len(response['context'])} contexts for full query") 