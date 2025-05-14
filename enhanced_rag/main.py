import os
from dotenv import load_dotenv

from src.self_route import SelfRoute
from src.long_context import LongContext
from src.vanilla_rag import VanillaRAG
from src.enhanced_rag import EnhancedRAG
from src.hyde import HyDE

load_dotenv()

parent_model_choices = ['openai', 'anthropic', 'gemini']
model_choices = ['gpt-4.1-mini', 'claude-3-5-sonnet-20240620', 'gemini-2.0-flash']
api_keys = [os.getenv('OPENAI_API_KEY'), os.getenv('CLAUDE_API_KEY'), os.getenv('GEMINI_API_KEY')]

MODEL_CHOICE = 0
PDF_PATH = 'tests/1682954_thesis_proposal.pdf'
QUERY = 'Which dataset or datasets will be used for evaluation?'

# # vanilla-RAG
def vanilla_rag(pdf_path: str, query: str):
    vanilla_rag = VanillaRAG(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = vanilla_rag.query(query)

    print('=' * 100)
    for context in results["context"]:
        print(f"\nSource page number: {context.metadata['page']}")
        print(context.page_content)
        print('=' * 100)

    print(f"Answer:\n{results['answer']}")

# long-context-RAG
def long_context_rag(pdf_path: str, query: str):
    long_context_rag = LongContext(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = long_context_rag.query(query)

    print('=' * 100)
    print(results)
    print('=' * 100)

    print(f"\n\nAnswer:\n{results.content}")

# enhanced-RAG
def enhanced_rag(pdf_path: str, query: str, preserve_order: bool = False):
    enhanced_rag = EnhancedRAG(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = enhanced_rag.query(query, preserve_order=preserve_order)

    print('=' * 100)
    print(results)
    print('=' * 100)
    print('\n\n')

    print(f"\n\nAnswer:\n{results['answer']}")


# self-route
def self_route(pdf_path: str, query: str):
    self_route = SelfRoute(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results, response_type = self_route.query(query)

    print('=' * 100)
    print(results)
    print('=' * 100)

    print(f"Response type: {response_type}")

    print(f"\n\nAnswer:\n{results['answer']}")

# Hypothetical Document Embedding
def hyde(pdf_path: str, query: str):
    hyde = HyDE(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = hyde.query(query)
    
    print('=' * 100)
    print(results)
    print('=' * 100)

    print(f"\n\nAnswer:\n{results['answer']}")

hyde(PDF_PATH, QUERY)
