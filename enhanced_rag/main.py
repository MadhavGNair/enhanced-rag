import os
from dotenv import load_dotenv

from self_route import SelfRoute
from long_context import LongContext
from vanilla_rag import VanillaRAG
from enhanced_rag import EnhancedRAG

load_dotenv()

parent_model_choices = ['openai', 'anthropic', 'gemini']
model_choices = ['gpt-4.1-mini', 'claude-3-5-sonnet-20240620', 'gemini-2.0-flash']
api_keys = [os.getenv('OPENAI_API_KEY'), os.getenv('CLAUDE_API_KEY'), os.getenv('GEMINI_API_KEY')]

MODEL_CHOICE = 0
PDF_PATH = 'tests/1682954_thesis_proposal.pdf'
QUERY = 'Who is the first supervisor of the thesis?'

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
