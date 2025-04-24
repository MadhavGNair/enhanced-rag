
# Enhanced RAG

Enhancing vanilla-RAG for sustainability report scoring.

## Description

As ESG scores in sustainability reporting have become more and more improtant, utilizing Large Language Models (LLMs) can automate and optimize this process. Various research have been done in the recent years on using RAG or Long-context LLMs for this purpose. The goal of this project is to leverage various advanced techniques and frameworks from various sub-fields of AI in order to improve vanilla-RAG. This improved version, named enhanced-RAG will then be put to test against long-context LLMs as well as vanilla-RAG, and a combination of the two called Self-Route. The test will help (1) identify the potential features from other research that can help with sustainability report scoring and (2) evalute the methods to select the best method for incorporating in the pipeline of a company.

Previous research have differentiated between long-context and short-context LLMs by their token limts. However, with recent advancements in the context window of models, pretty much every SOTA model has long-context capabilities. Therefore, the conditions tested in this research is defined as follows:

1. control group - the passage relevant to answering the query is provided directly to the LLM
2. enhanced-RAG - the query is used to search a database of passages and top *k* is retrieved and used to prompt the LLM
3. self-route - similar to 2 but LLM can choose to not answer which then triggers long-context prompting
4. long-context - the entire PDF is passed as input along with the query

## Dataset

Two datasets will be used to evaluate the aforementioned conditions, namely

1. [NEPAQuAD1.0](https://www.kaggle.com/competitions/llm-for-environmental-review/data) - a dataset of Q&A pairs from the NEPA documents. Contains *context*, which is the passage relevant to the question, *question*, which is the query, and *ground truth answer*, which is the correct answer to the query.
2. [PromiseEval](https://drive.google.com/drive/folders/1wWwo5DBY2qFj2KSEqjkjinuK5CB5ku5K) - a dataset of promises that has been found in company reports. These promises were hand-extracted to fall into one of the ESG pillars. Contains *url* to PDF, *data*, which is the passage, and *promise status*, which is a boolean that indicates whether the passage has a promise or not. These promises can be cross-checked with factors or query from sustainability reports to be used as ground truth for testing. So this dataset requries additional processing in order to work with the pipeline of this research.

NOTE: the data within the datasets where not collected or altered by me.
