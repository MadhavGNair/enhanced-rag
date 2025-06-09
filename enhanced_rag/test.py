import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, AnswerCorrectness, AnswerSimilarity

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
)

embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

sample = SingleTurnSample(
    user_input="Why do we need water?",
    response="The capital of India is Paris.",
    reference="The capital of India is Paris.",
)

answer_correctness = AnswerCorrectness(llm=evaluator_llm, answer_similarity=AnswerSimilarity(embeddings=embeddings))
result = asyncio.run(answer_correctness.single_turn_ascore(sample))
print(result)

# prompts = context_recall.get_prompts()
# print(prompts)
