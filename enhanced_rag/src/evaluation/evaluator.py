from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerCorrectness, AnswerSimilarity, Faithfulness, LLMContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate


class Evaluator:
    def __init__(
        self,
        framework: str,
        response_dict: dict
    ):
        self.framework = framework
        self.response_dict = response_dict
        self.query_data = response_dict["question"]
        self.response = response_dict["answer"]
        if not (self.framework == "long_context" or (self.framework == "self_route" and response_dict.get("response_type") == "long_context")):
            self.contexts = [context["page_content"] for context in response_dict["retrieved_contexts"]]
        self.ground_truth = response_dict["groundtruth_answer"]

    def __curate_entries(self):
        if self.framework == "long_context" or (self.framework == "self_route" and self.response_dict.get("response_type") == "long_context"):
            answer_correctness_entry = SingleTurnSample(
                user_input=self.query_data,
                response=self.response,
                reference=self.ground_truth,
            )
            return answer_correctness_entry
        else:
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
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
        answer_correctness_score = None
        context_recall_score = None
        faithfulness_score = None

        if self.framework == "long_context" or (self.framework == "self_route" and self.response_dict.get("response_type") == "long_context"):
            # answer correctness
            answer_correctness_entry = self.__curate_entries()
            answer_correctness = AnswerCorrectness(llm=evaluator_llm, answer_similarity=AnswerSimilarity(embeddings=embeddings))
            answer_correctness_score = await answer_correctness.single_turn_ascore(answer_correctness_entry)
        else:
            context_recall_entry, answer_correctness_entry, faithfulness_entry = self.__curate_entries()

            answer_correctness = AnswerCorrectness(llm=evaluator_llm, answer_similarity=AnswerSimilarity(embeddings=embeddings))
            answer_correctness_score = await answer_correctness.single_turn_ascore(answer_correctness_entry)

            # context recall
            context_recall = LLMContextRecall(llm=evaluator_llm)
            context_recall_score = await context_recall.single_turn_ascore(context_recall_entry)

            # faithfulness
            faithfulness = Faithfulness(llm=evaluator_llm)
            faithfulness_score = await faithfulness.single_turn_ascore(faithfulness_entry)

        return answer_correctness_score, context_recall_score, faithfulness_score
