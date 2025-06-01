import os
from dotenv import load_dotenv
import pandas as pd

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
ROOT_PATH = r'D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\pdfs_train'
# PDF_PATH = 'tests/1682954_thesis_proposal.pdf'
# QUERY = 'Which dataset or datasets will be used for evaluation?'

# # vanilla-RAG
def vanilla_rag(pdf_path: str, query: str):
    vanilla_rag = VanillaRAG(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = vanilla_rag.query(query)

    # print('=' * 100)
    # for context in results["context"]:
    #     print(f"\nSource page number: {context.metadata['page']}")
    #     print(context.page_content)
    #     print('=' * 100)

    # print(f"Answer:\n{results['answer']}")
    return results

# long-context-RAG
def long_context_rag(pdf_path: str, query: str):
    long_context_rag = LongContext(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = long_context_rag.query(query)

    # print('=' * 100)
    # print(results)
    # print('=' * 100)

    # print(f"\n\nAnswer:\n{results.content}")
    return results

# enhanced-RAG
def enhanced_rag(pdf_path: str, query: str, preserve_order: bool = False):
    enhanced_rag = EnhancedRAG(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = enhanced_rag.query(query, preserve_order=preserve_order)

    # print('=' * 100)
    # print(results)
    # print('=' * 100)
    # print('\n\n')

    # print(f"\n\nAnswer:\n{results['answer']}")
    return results


# self-route
def self_route(pdf_path: str, query: str):
    self_route = SelfRoute(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results, response_type = self_route.query(query)

    # print('=' * 100)
    # print(results)
    # print('=' * 100)

    # print(f"Response type: {response_type}")

    # print(f"\n\nAnswer:\n{results['answer']}")
    return results

# Hypothetical Document Embedding
def hyde(pdf_path: str, query: str):
    hyde = HyDE(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    results = hyde.query(query)
    
    # print('=' * 100)
    # print(results)
    # print('=' * 100)

    # print(f"\n\nAnswer:\n{results['answer']}")
    return results

def display_results(question: str, groundtruth_answer: str, groundtruth_context: str, results: dict, method_name: str):
    print("\n" + "="*100)
    print(f"METHOD: {method_name.upper()}")
    print("="*100)
    
    print("\nQUESTION:")
    print("-"*50)
    print(question)
    
    print("\nGROUNDTRUTH ANSWER:")
    print("-"*50)
    print(groundtruth_answer)
    
    print("\nGROUNDTRUTH CONTEXT:")
    print("-"*50)
    print(groundtruth_context)
    
    print("\nMODEL'S ANSWER:")
    print("-"*50)
    print(results['answer'])
    
    if 'context' in results:
        print("\nRETRIEVED CONTEXTS:")
        print("-"*50)
        for i, context in enumerate(results['context'], 1):
            print(f"\nContext {i}:")
            print(f"Source page: {context.metadata['page']}")
            print(context.page_content)
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    # load the data
    data = pd.read_csv(r'D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\NEPAQuAD1_train.csv')

    sample_data = data.iloc[0]

    question = sample_data['question']
    groundtruth_answer = sample_data['groundtruth_answer']
    groundtruth_context = sample_data['context']
    pdf_path = os.path.join(ROOT_PATH, sample_data['EIS_filename'] + '.pdf')

    # Run and display results for each method
    results = vanilla_rag(pdf_path, question)
    display_results(question, groundtruth_answer, groundtruth_context, results, "vanilla_rag")
    
    # Uncomment to test other methods
    # results = long_context_rag(pdf_path, question)
    # display_results(question, groundtruth_answer, groundtruth_context, results, "long_context_rag")
    
    # results = enhanced_rag(pdf_path, question)
    # display_results(question, groundtruth_answer, groundtruth_context, results, "enhanced_rag")
    
    # results = self_route(pdf_path, question)
    # display_results(question, groundtruth_answer, groundtruth_context, results, "self_route")
    
    # results = hyde(pdf_path, question)
    # display_results(question, groundtruth_answer, groundtruth_context, results, "hyde")

