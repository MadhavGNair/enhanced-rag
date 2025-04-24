from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class VanillaRAG:
    def __init__(self, pdf_path: str, model_name: str, api_key: str, parent_model: str = 'openai'):
        # load and split the pdf
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()

        # initialize the model
        if parent_model == 'openai':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
        elif parent_model == 'anthropic':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatAnthropic(model_name=self.model_name, api_key=self.api_key)
        elif parent_model == 'gemini':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatVertexAI(model_name=self.model_name, api_key=self.api_key)
        else:
            raise ValueError(f"Invalid parent model: {parent_model}")

        # initialize the system prompt
        self.system_prompt = """
        You are a helpful assistant that can answer questions about the document. Use the provided context to answer the question. If you don't know the answer, say "I don't know". Be concise and to the point.\n\n{context}
        """

    def __generate_embeddings(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(self.docs)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        return vector_store.as_retriever()

    def query(self, query: str):
        retriever = self.__generate_embeddings()
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}"),
        ])
        q_and_a_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
        )
        chain = create_retrieval_chain(
            retriever, q_and_a_chain
        )
        return chain.invoke({"input": query})
