from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import AnswerRelevancy, Faithfulness, FactualCorrectness


class Evaluator:
    def __init__(self, query_data: list[dict], response: list[dict], contexts: list[dict], ground_truth: list[dict]):
        self.query_data = query_data
        self.response = response
        self.contexts = contexts
        self.ground_truth = ground_truth

    def __curate_dataset(self):
        dataset = []
        for query, response, context, ground_truth in zip(self.query_data, self.response, self.contexts, self.ground_truth):
            dataset.append({
                "query": query,
                "response": response,
                "context": context,
                "ground_truth": ground_truth
            })
        return EvaluationDataset.from_list(dataset)

    def evaluate(self):
        dataset = self.__curate_dataset()
        results = evaluate(dataset, metrics=[AnswerRelevancy(), Faithfulness(), FactualCorrectness()])
        return results
