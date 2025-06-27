import asyncio
import os
import re
import json
import time
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
from src.evaluation.evaluator import Evaluator
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

data = pd.read_csv(
    r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\NEPAQuAD1_train.csv"
)

# pdf_names = data["EIS_filename"].unique()
pdf_names = ['eis_Continental_United_States_Interceptor_Site']

for pdf_name in tqdm(pdf_names, desc="Processing PDFs"):
    # define pdf path
    pdf_path = os.path.join(ROOT_PATH, pdf_name + ".pdf")

    # initialize frameworks
    vanilla_rag = VanillaRAG(pdf_path, model_choices[MODEL_CHOICE], api_keys[MODEL_CHOICE], parent_model_choices[MODEL_CHOICE])

    # get subset of data for the current pdf
    subset_data = data[data["EIS_filename"] == pdf_name]

    # query the data
    for index, row in tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"Processing questions for {pdf_name}", leave=False):
        # Skip if this question has already been processed
        if index == 3:
            break

        row_id = row["ID"]
        question = row["question"]
        question_type = row["question_type"]
        groundtruth_answer = row["groundtruth_answer"]
        groundtruth_context = row["context"]

        # Run frameworks and collect responses
        start_time = time.time()
        vanilla_rag_response = vanilla_rag.query(question)
        vanilla_time = time.time() - start_time
        
        print(f'answer: {vanilla_rag_response["answer"]}')
        print(f'context: {vanilla_rag_response["context"]}')
        print('-'*100)