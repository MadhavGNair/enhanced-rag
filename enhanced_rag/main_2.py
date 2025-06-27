import asyncio
import os
import re
import json
import time
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
from src.evaluation.evaluator import Evaluator
from src.enhanced_rag import EnhancedRAG
from src.hyde import HyDE
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

MODEL_CHOICE = 2
ROOT_PATH = r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\pdfs_train"

def initialize_frameworks(pdf_path):
    vanilla_rag = VanillaRAG(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )
    enhanced_rag = EnhancedRAG(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )
    self_route = SelfRoute(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )
    hyde = HyDE(
        pdf_path,
        model_choices[MODEL_CHOICE],
        api_keys[MODEL_CHOICE],
        parent_model_choices[MODEL_CHOICE],
    )

    return vanilla_rag, enhanced_rag, self_route, hyde


async def evaluate_results(response_dict: dict, framework: str):
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
    pattern = r"^(yes|no)(?=[,.\s:;\-•\n]|$)"
    match = re.match(pattern, cleaned_response)

    if match:
        return match.group(1)
    else:
        return response


def load_existing_results():
    results = {
        "vanilla_rag": [],
        "enhanced_rag": [],
        "self_route": [],
        "hyde": [],
    }
    
    # Create results/new_results directory if it doesn't exist
    if not os.path.exists("results/new_results"):
        os.makedirs("results/new_results")
        return results
    
    # Load existing results if they exist
    for key in results.keys():
        file_path = f"results/new_results/{key}.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                results[key] = json.load(f)
    
    return results

def save_results(results):
    for key, value in results.items():
        serializable_value = convert_to_serializable(value)
        with open(f"results/new_results/{key}.json", "w", encoding="utf-8") as f:
            json.dump(serializable_value, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # load the data
    data = pd.read_csv(
        r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\NEPAQuAD1_train.csv"
    )

    # Load existing results or initialize new ones
    results = load_existing_results()

    pdf_names = data["EIS_filename"].unique()

    for pdf_name in tqdm(pdf_names, desc="Processing PDFs"):
        # define pdf path
        pdf_path = os.path.join(ROOT_PATH, pdf_name + ".pdf")

        # initialize frameworks
        vanilla_rag, enhanced_rag, self_route, hyde = initialize_frameworks(pdf_path)

        # get subset of data for the current pdf
        subset_data = data[data["EIS_filename"] == pdf_name]

        # query the data
        for index, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"Processing questions for {pdf_name}", leave=False):
            # Skip if this question has already been processed
            if any(index == result["index"] for result in results["vanilla_rag"]):
                continue

            row_id = row["ID"]
            question = row["question"]
            question_type = row["question_type"]
            groundtruth_answer = row["groundtruth_answer"]
            groundtruth_context = row["context"]

            # Run frameworks and collect responses
            start_time = time.time()
            vanilla_rag_response = vanilla_rag.query(question)
            vanilla_time = time.time() - start_time
            
            start_time = time.time()
            enhanced_rag_response = enhanced_rag.query(question)
            enhanced_time = time.time() - start_time
            
            start_time = time.time()
            self_route_response, response_type = self_route.query(question)
            self_route_time = time.time() - start_time
            
            start_time = time.time()
            hyde_response = hyde.query(question)
            hyde_time = time.time() - start_time

            # define structure for the results
            vanilla_rag_results = {
                "index": index,
                "ID": row_id,
                "pdf_name": pdf_name,
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
                "execution_time": vanilla_time,
            }

            enhanced_rag_results = {
                "index": index,
                "ID": row_id,
                "pdf_name": pdf_name,
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
                "execution_time": enhanced_time,
            }

            if response_type == "vanilla_rag":
                self_route_results = {
                    "index": index,
                    "ID": row_id,
                    "pdf_name": pdf_name,
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
                    "execution_time": self_route_time,
                }
            elif response_type == "long_context":
                self_route_results = {
                    "index": index,
                    "ID": row_id,
                    "pdf_name": pdf_name,
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
                    "execution_time": self_route_time,
                }

            hyde_results = {
                "index": index,
                "ID": row_id,
                "pdf_name": pdf_name,
                "question": question,
                "question_type": question_type,
                "answer": hyde_response["answer"],
                "hypothetical_document": hyde_response["hypothetical_document"],
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
                "execution_time": hyde_time,
            }

            # post-process the responses
            if question_type == "closed":
                vanilla_rag_results["answer"] = extract_yes_no(vanilla_rag_results["answer"])
                enhanced_rag_results["answer"] = extract_yes_no(enhanced_rag_results["answer"])
                self_route_results["answer"] = extract_yes_no(self_route_results["answer"])
                hyde_results["answer"] = extract_yes_no(hyde_results["answer"])

            eval_results = {
                "vanilla_rag": vanilla_rag_results,
                "enhanced_rag": enhanced_rag_results,
                "self_route": self_route_results,
                "hyde": hyde_results,
            }

            # evaluate the results
            for key, value in eval_results.items():
                response_dict = value
                framework = key
                answer_correctness_score, context_recall_score, faithfulness_score = asyncio.run(evaluate_results(response_dict, framework))
                eval_results[key]["evaluation"]["answer_correctness"] = answer_correctness_score
                eval_results[key]["evaluation"]["context_recall"] = context_recall_score
                eval_results[key]["evaluation"]["faithfulness"] = faithfulness_score

            # save the results after each question
            results["vanilla_rag"].append(vanilla_rag_results)
            results["enhanced_rag"].append(enhanced_rag_results)
            results["self_route"].append(self_route_results)
            results["hyde"].append(hyde_results)
            
            # Save results after each question
            save_results(results)
            print(f"Saved results for question {row_id}")

    print("All questions processed and results saved.")
