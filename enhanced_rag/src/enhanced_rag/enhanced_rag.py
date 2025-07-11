import hashlib
from typing import List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EnhancedRAG:
    def __init__(
        self, pdf_path: str, model_name: str, api_key: str, parent_model: str = "openai"
    ):
        # load and split the pdf
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load_and_split()

        # initialize the embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.semantic_retriever = self.__initialize_semantic_retriever()
        self.bm25_retriever = self.__initialize_BM25_retriever()
        
        # initialize the model
        if parent_model == "openai":
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
        elif parent_model == "anthropic":
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatAnthropic(model=self.model_name, api_key=self.api_key)
        elif parent_model == "gemini":
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, api_key=self.api_key
            )
        else:
            raise ValueError(f"Invalid parent model: {parent_model}")

        # initialize the system prompt
        self.system_prompt = """
        You are a helpful assistant that can answer questions about the document. Use the provided context to answer the question. If the question is a yes or no question, answer with only "yes" or "no" without any other text. Be concise and to the point.\n\n{context}
        """

    def __initialize_semantic_retriever(self):
        """
        Initialize the semantic retriever.

        Returns:
            vector_store.as_retriever(): A retriever object that can be used to retrieve documents from the vector store.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.docs)
        
        # create a unique collection name to prevent interference between frameworks
        pdf_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()[:8]
        collection_name = f"enhanced_rag_{pdf_hash}"
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
        )
        return vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def __initialize_BM25_retriever(self):
        """
        Initialize the BM25 retriever.

        Returns:
            bm25_retriever: A retriever object that can be used to retrieve documents using BM25.
        """
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 3

        return bm25_retriever

    def query(self, query: str, k: int = 3, preserve_order: bool = False):
        """
        Query the RAG model with option to preserve document order.

        Args:
            query (str): The query to answer.
            k (int): The number of documents to retrieve (default: 3).
            preserve_order (bool): If True, reorder chunks by page number before passing to LLM (default: False).

        Returns:
            dict: The LLM response with answer and retrieved context.
        """
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            weights=[0.6, 0.4],
        )

        # get documents from the ensemble retriever
        retrieved_docs = ensemble_retriever.invoke(query)

        # limit to top k
        retrieved_docs = retrieved_docs[:k]

        # sort by page number if preserve_order is True
        if preserve_order:
            retrieved_docs = sorted(
                retrieved_docs,
                key=lambda doc: (
                    doc.metadata.get("page", 0),
                    (
                        doc.metadata.get("position", 0)
                        if "position" in doc.metadata
                        else 0
                    ),
                ),
            )

        # create a context for the LLM from the retrieved documents
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # create the prompt
        system_prompt_with_context = self.system_prompt.replace("{context}", context)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_with_context),
                ("user", "{input}"),
            ]
        )

        # use the LLM to answer based on the retrieved context
        chain = prompt | self.llm

        # return both the answer and the documents for analysis
        result = {
            "answer": chain.invoke({"input": query}),
            "source_documents": retrieved_docs,
        }

        return result
