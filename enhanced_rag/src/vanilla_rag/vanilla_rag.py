import hashlib
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VanillaRAG:
    def __init__(
        self, pdf_path: str, model_name: str, api_key: str, parent_model: str = "openai", system_prompt: str = None, collection_prefix: str = "vanilla_rag"
    ):
        # load and split the pdf
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load_and_split()
        
        # Store the API keys for embeddings
        self.api_key = api_key
        self.parent_model = parent_model
        self.collection_prefix = collection_prefix
        
        self.retriever = self.__generate_embeddings()

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
        if system_prompt is None:
            self.system_prompt = """
            You are a helpful assistant that can answer questions about the document. Use the provided context to answer the question. Answer only based on the context. If the question is a yes or no question, answer with only "yes" or "no" without any other text. Be concise and to the point.\n\n{context}
            """
        else:
            self.system_prompt = system_prompt

    def __generate_embeddings(self):
        """
        Generate embeddings for the documents.

        Returns:
            vector_store.as_retriever(): A retriever object that can be used to retrieve documents from the vector store.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.docs)
        
        # Always use OpenAI API key for embeddings, regardless of LLM choice
        openai_api_key = self.api_key if self.parent_model == "openai" else os.getenv("OPENAI_API_KEY")
        
        # create a unique collection name to prevent interference between frameworks
        pdf_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()[:8]
        collection_name = f"{self.collection_prefix}_{pdf_hash}"
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_api_key
            ),
            collection_name=collection_name,
        )
        return vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def query(self, query: str):
        """
        Query the RAG model.

        Args:
            query (str): The query to answer.

        Returns:
            chain.invoke({"input": query}): The answer to the query.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "{input}"),
            ]
        )
        q_and_a_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
        )
        chain = create_retrieval_chain(self.retriever, q_and_a_chain)
        return chain.invoke({"input": query})
