import json
from pprint import pprint
import os
from tqdm import tqdm

frameworks = ['enhanced_rag', 'hyde', 'long_context', 'self_route', 'vanilla_rag']

# Create the target directory if it doesn't exist
os.makedirs('results/gpt_evaluation', exist_ok=True)

for framework in tqdm(frameworks, desc="Processing frameworks", position=0):
    with open(f'results/{framework}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_results = []
    for item in tqdm(data, desc=f"Processing {framework}", position=1, leave=False):
        result = {
            "index": item['index'],
            "ID": item['ID'],
            "evaluation": item['evaluation']
        }
        processed_results.append(result)
    
    # Save the processed results to a new JSON file
    output_file = f'results/gpt_evaluation/{framework}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print(f'Saved processed results to {output_file}')

        
