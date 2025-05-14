from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


class HyDE:
    def __init__(self, pdf_path: str, model_name: str, api_key: str, parent_model: str = 'openai'):
        # load and split the pdf
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()

        # initialize the model
        self.parent_model = parent_model
        if parent_model == 'openai':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatOpenAI(model_name=self.model_name, api_key=self.api_key)
        elif parent_model == 'anthropic':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatAnthropic(model=self.model_name, api_key=self.api_key)
        elif parent_model == 'gemini':
            self.model_name = model_name
            self.api_key = api_key
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, api_key=self.api_key)
        else:
            raise ValueError(f"Invalid parent model: {parent_model}")

        # initialize the system prompt
        self.system_prompt = """
        You are a helpful assistant that can answer questions about the document. Use the provided context to answer the question. Answer only based on the context. If you cannot answer based on the context, respond with "Out of context". Be concise and to the point.\n\n{context}
        """

        self.hyde_prompt = """
        Please generate a concise hypothetical answer to the question: {question}.
        """

    def __generate_embeddings(self):
        """
        Generate embeddings for the documents.

        Returns:
            vector_store:  Chroma vector store.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(self.docs)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        return vector_store

    def __hyde(self, query: str) -> str:
        """
        Generates a hypothetical document based on the query using the LLM.

        Args:
            query (str): The query to generate a hypothetical document for.

        Returns:
            str: The hypothetical document.
        """
        hyde_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a helpful assistant that generates hypothetical responses to a given query."),
            HumanMessagePromptTemplate.from_template(self.hyde_prompt)
        ])
        hyde_prompt = hyde_prompt_template.format_messages(question=query)

        hypothetical_document = self.llm.invoke(hyde_prompt).content
        return hypothetical_document

    def query(self, query: str):
        """
        Query the RAG model using HyDE.

        Args:
            query (str): The query to answer.

        Returns:
            chain.invoke({"input": query}): The answer to the query.
        """
        vector_store = self.__generate_embeddings()
        hypothetical_document = self.__hyde(query)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})

        retrieved_documents = retriever.invoke(hypothetical_document)

        print(f"Hypothetical document: \n\n{hypothetical_document}\n\n")
        for doc in retrieved_documents:
            print(f"Retrieved document: \n\n{doc.page_content}\n\n")
        print('=' * 100)
        print('\n\n')

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
        return chain.invoke({"input": query, "context": retrieved_documents})