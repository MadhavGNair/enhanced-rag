from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerCorrectness, AnswerSimilarity, Faithfulness, LLMContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate


class Evaluator:
    def __init__(
        self,
        query_data: list[dict],
        response: list[dict],
        contexts: list[dict],
        ground_truth: list[dict],
    ):
        self.query_data = query_data
        self.response = response
        self.contexts = [context.page_content for context in contexts]
        self.ground_truth = ground_truth

    def __curate_entries(self):
        context_recall_entry = SingleTurnSample(
            user_input=self.query_data,
            retrieved_contexts=self.contexts,
            reference=self.ground_truth,
        )

        answer_correctness_entry = SingleTurnSample(
            user_input=self.query_data,
            response=self.response,
            reference=self.ground_truth,
        )

        faithfulness_entry = SingleTurnSample(
            user_input=self.query_data,
            response=self.response,
            retrieved_contexts=self.contexts,
        )

        return context_recall_entry, answer_correctness_entry, faithfulness_entry

    async def evaluate(self):
        results = {
            "answer_correctness": 0,
            "context_recall": 0,
            "faithfulness": 0,
        }
        context_recall_entry, answer_correctness_entry, faithfulness_entry = self.__curate_entries()
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        # answer correctness
        answer_correctness = AnswerCorrectness(llm=evaluator_llm, answer_similarity=AnswerSimilarity(embeddings=embeddings))
        results["answer_correctness"] = await answer_correctness.single_turn_ascore(answer_correctness_entry)

        # context recall
        context_recall = LLMContextRecall(llm=evaluator_llm)
        results["context_recall"] = await context_recall.single_turn_ascore(context_recall_entry)

        # faithfulness
        faithfulness = Faithfulness(llm=evaluator_llm)
        results["faithfulness"] = await faithfulness.single_turn_ascore(faithfulness_entry)

        return results
