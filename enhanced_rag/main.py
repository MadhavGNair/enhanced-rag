import asyncio
import os
import re
import json

import pandas as pd
from dotenv import load_dotenv
from src.enhanced_rag import EnhancedRAG
from src.evaluation.evaluator import Evaluator
from src.hyde import HyDE
from src.long_context import LongContext
from src.self_route import SelfRoute
from src.vanilla_rag import VanillaRAG

load_dotenv()

parent_model_choices = ["openai", "anthropic", "gemini"]
model_choices = ["gpt-4.1-mini", "claude-3-5-sonnet-20240620", "gemini-2.0-flash"]
api_keys = [
    os.getenv("OPENAI_API_KEY"),
    os.getenv("CLAUDE_API_KEY"),
    os.getenv("GEMINI_API_KEY"),
]

MODEL_CHOICE = 0
ROOT_PATH = r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\pdfs_train"
# PDF_PATH = 'tests/1682954_thesis_proposal.pdf'
# QUERY = 'Which dataset or datasets will be used for evaluation?'


# # vanilla-RAG
def vanilla_rag(pdf_path: str, query: str):
    vanilla_rag = VanillaRAG(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    results = vanilla_rag.query(query)
    return results


# long-context-RAG
def long_context_rag(pdf_path: str, query: str):
    long_context_rag = LongContext(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    results = long_context_rag.query(query)
    return results


# enhanced-RAG
def enhanced_rag(pdf_path: str, query: str, preserve_order: bool = False):
    enhanced_rag = EnhancedRAG(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    results = enhanced_rag.query(query, preserve_order=preserve_order)
    return results


# self-route
def self_route(pdf_path: str, query: str):
    self_route = SelfRoute(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    results, response_type = self_route.query(query)
    return results, response_type


# Hypothetical Document Embedding
def hyde(pdf_path: str, query: str):
    hyde = HyDE(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    results = hyde.query(query)
    return results

async def evaluate_results(
    response_dict: dict,
    framework: str
):
    evaluator = Evaluator(framework, response_dict)
    return await evaluator.evaluate()


def convert_to_serializable(obj):
    if hasattr(obj, "page_content"):
        return {"page_content": obj.page_content}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def extract_yes_no(response):
    cleaned_response = response.strip().lower()
    pattern = r'^(yes|no)(?=[,.\s:;\-•\n]|$)'
    match = re.match(pattern, cleaned_response)
    
    if match:
        return match.group(1)
    else:
        return response


if __name__ == "__main__":
    # load the data
    data = pd.read_csv(
        r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\NEPAQuAD1_train.csv"
    )

    results = {
        "vanilla_rag": [],
        "long_context": [],
        "enhanced_rag": [],
        "self_route": [],
        "hyde": [],
    }
    
    for index in range(len(data)):

        if index == 5:
            break

        print(f"Processing index {index}")

        sample_data = data.iloc[index]
            
        question = sample_data["question"]
        question_type = sample_data["question_type"]
        groundtruth_answer = sample_data["groundtruth_answer"]
        groundtruth_context = sample_data["context"]
        pdf_path = os.path.join(ROOT_PATH, sample_data["EIS_filename"] + ".pdf")

        # run and save results for each framework
        vanilla_rag_response = vanilla_rag(pdf_path, question)
        long_context_response = long_context_rag(pdf_path, question)
        enhanced_rag_response = enhanced_rag(pdf_path, question)
        self_route_response, response_type = self_route(pdf_path, question)
        hyde_response = hyde(pdf_path, question)

        # define structure for the results
        vanilla_rag_results = {
            "index": index,
            "pdf_name": sample_data["EIS_filename"],
            "question": question,
            "question_type": question_type,
            "answer": vanilla_rag_response["answer"],
            "groundtruth_answer": groundtruth_answer,
            "retrieved_contexts": [
                {
                    "page_number": context.metadata["page"],
                    "page_content": context.page_content,
                }
                for context in vanilla_rag_response["context"]
            ],
            "groundtruth_context": groundtruth_context,
            "evaluation": {
                "answer_correctness": None,
                "context_recall": None,
                "faithfulness": None,
            },
        }

        long_context_results = {
            "index": index,
            "pdf_name": sample_data["EIS_filename"],
            "question": question,
            "question_type": question_type,
            "answer": long_context_response.content,
            "groundtruth_answer": groundtruth_answer,
            "evaluation": {
                "answer_correctness": None,
                "context_recall": None,
                "faithfulness": None,
            },
        }

        enhanced_rag_results = {
            "index": index,
            "pdf_name": sample_data["EIS_filename"],
            "question": question,
            "question_type": question_type,
            "answer": enhanced_rag_response["answer"].content,
            "groundtruth_answer": groundtruth_answer,
            "retrieved_contexts": [
                {
                    "page_number": context.metadata["page"],
                    "page_content": context.page_content,
                }
                for context in enhanced_rag_response["source_documents"]
            ],
            "groundtruth_context": groundtruth_context,
            "evaluation": {
                "answer_correctness": None,
                "context_recall": None,
                "faithfulness": None,
            },
        }

        if response_type == "vanilla_rag":
            self_route_results = {
                "index": index,
                "pdf_name": sample_data["EIS_filename"],
                "question": question,
                "question_type": question_type,
                "response_type": response_type,
                "answer": self_route_response["answer"],
                "groundtruth_answer": groundtruth_answer,
                "retrieved_contexts": [
                    {
                        "page_number": context.metadata["page"],
                        "page_content": context.page_content,
                    }
                    for context in self_route_response["context"]
                ],
                "groundtruth_context": groundtruth_context,
                "evaluation": {
                    "answer_correctness": None,
                    "context_recall": None,
                    "faithfulness": None,
                },
            }
        elif response_type == "long_context":
            self_route_results = {
                "index": index,
                "pdf_name": sample_data["EIS_filename"],
                "question": question,
                "question_type": question_type,
                "response_type": response_type,
                "answer": self_route_response.content,
                "groundtruth_answer": groundtruth_answer,
                "evaluation": {
                    "answer_correctness": None,
                    "context_recall": None,
                    "faithfulness": None,
                },
            }

        hyde_results = {
            "index": index,
            "pdf_name": sample_data["EIS_filename"],
            "question": question,
            "question_type": question_type,
            "answer": hyde_response["answer"],
            "groundtruth_answer": groundtruth_answer,
            "retrieved_contexts": [
                {
                    "page_number": context.metadata["page"],
                    "page_content": context.page_content,
                }
                for context in hyde_response["context"]
            ],
            "groundtruth_context": groundtruth_context,
            "evaluation": {
                "answer_correctness": None,
                "context_recall": None,
                "faithfulness": None,
            },
        }

        # post-process the responses
        if question_type == "closed":
            vanilla_rag_results["answer"] = extract_yes_no(vanilla_rag_results["answer"])
            long_context_results["answer"] = extract_yes_no(long_context_results["answer"])
            enhanced_rag_results["answer"] = extract_yes_no(enhanced_rag_results["answer"])
            self_route_results["answer"] = extract_yes_no(self_route_results["answer"])
            hyde_results["answer"] = extract_yes_no(hyde_results["answer"])

        eval_results = {
            "vanilla_rag": vanilla_rag_results,
            "long_context": long_context_results,
            "enhanced_rag": enhanced_rag_results,
            "self_route": self_route_results,
            "hyde": hyde_results,
        }

        # evaluate the results
        for key, value in eval_results.items():
            print(f"    Evaluating {key}...")
            response_dict = value
            framework = key
            answer_correctness_score, context_recall_score, faithfulness_score = asyncio.run(evaluate_results(response_dict, framework))
            eval_results[key]["evaluation"]["answer_correctness"] = answer_correctness_score
            eval_results[key]["evaluation"]["context_recall"] = context_recall_score
            eval_results[key]["evaluation"]["faithfulness"] = faithfulness_score

        # save the results
        results["vanilla_rag"].append(vanilla_rag_results)
        results["long_context"].append(long_context_results)
        results["enhanced_rag"].append(enhanced_rag_results)
        results["self_route"].append(self_route_results)
        results["hyde"].append(hyde_results)

    # create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    for key, value in results.items():
        serializable_value = convert_to_serializable(value)
        with open(f"results/{key}.json", "w", encoding="utf-8") as f:
            json.dump(serializable_value, f, ensure_ascii=False, indent=4)
        print(f"Results for {key} saved to results/{key}.json")
