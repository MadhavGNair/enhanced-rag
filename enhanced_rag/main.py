import os
from dotenv import load_dotenv

from self_route import SelfRoute
from long_context import LongContext
from vanilla_rag import VanillaRAG
from enhanced_rag import EnhancedRAG

load_dotenv()

model_choices = ['openai', 'anthropic', 'gemini']

vanilla_rag = VanillaRAG('tests/1682954_thesis_proposal.pdf', 'gpt-4o-mini', os.getenv('OPENAI_API_KEY'), model_choices[0])

results = vanilla_rag.query('What is name of the author of the thesis?')


print(results["context"][0].page_content)

print('-' * 100)
print('\n\n')

print(results["context"][0].metadata)

print('-' * 100)
print('\n\n')

print(results['answer'])
