from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LongContext:
    def __init__(
        self, pdf_path: str, model_name: str, api_key: str, parent_model: str = "openai"
    ):
        # load and split the pdf
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()

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
        You are a helpful assistant that can answer questions about the document. Use the provided context to answer the question. Answer only based on the context. If you cannot answer based on the context, respond with "Out of context". Be concise and to the point.\n\n{context}
        """

    def __concate_docs(self):
        """
        Concatenate the documents.

        Returns:
            vector_store.as_retriever(): A retriever object that can be used to retrieve documents from the vector store.
        """
        context = ""
        for doc in self.docs:
            context += doc.page_content
        return context

    def query(self, query: str):
        """
        Query the RAG model.

        Args:
            query (str): The query to answer.

        Returns:
            chain.invoke({"input": query, "context": context}): The answer to the query.
        """
        context = self.__concate_docs()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "{input}"),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"input": query, "context": context})
