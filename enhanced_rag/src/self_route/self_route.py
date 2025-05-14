from src.vanilla_rag import VanillaRAG
from src.long_context import LongContext


class SelfRoute:
    def __init__(self, pdf_path: str, model_name: str, api_key: str, parent_model: str = 'openai'):
        self.vanilla_rag = VanillaRAG(pdf_path, model_name, api_key, parent_model)
        self.long_context = LongContext(pdf_path, model_name, api_key, parent_model)

    def query(self, query: str):
        """
        Query the RAG model.

        Args:
            query (str): The query to answer.

        Returns:
            rag_result: The answer to the query.
            OR
            long_context_result: The answer to the query from the long context.
        """
        rag_result = self.vanilla_rag.query(query)

        if rag_result['answer'] == 'Out of context':
            long_context_result = self.long_context.query(query)
            return long_context_result, 'long_context'
        else:
            return rag_result, 'vanilla_rag'


