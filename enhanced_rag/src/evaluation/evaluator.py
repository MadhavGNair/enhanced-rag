from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy, Faithfulness, LLMContextRecall
from langchain_openai import ChatOpenAI


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

    def __curate_entry(self):
        entry = SingleTurnSample(
            user_input=self.query_data,
            response=self.response,
            reference=self.ground_truth,
            retrieved_contexts=self.contexts,
        )
        return entry

    async def evaluate(self):
        results = {
            "context_recall": 0,
            "faithfulness": 0,
            "answer_accuracy": 0,
        }
        entry = self.__curate_entry()
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

        # context recall
        context_recall = LLMContextRecall(llm=evaluator_llm)
        results["context_recall"] = await context_recall.single_turn_ascore(entry)

        # faithfulness
        faithfulness = Faithfulness(llm=evaluator_llm)
        results["faithfulness"] = await faithfulness.single_turn_ascore(entry)

        # answer correctness
        answer_accuracy = AnswerAccuracy(llm=evaluator_llm)
        results["answer_accuracy"] = await answer_accuracy.single_turn_ascore(entry)
        return results
