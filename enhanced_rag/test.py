import pandas as pd

df = pd.read_csv('results/statistical_analysis/raw_data.csv')

df = df[df['framework'] == 'enhanced_rag'].drop(columns=['response_type'])

# Convert float columns to strings with double quotes
for col in ['answer_correctness', 'context_recall', 'faithfulness']:
    df[col] = df[col].astype(str).str.replace(r'^(\d+\.?\d*)$', r'"\1"', regex=True)

# Save to CSV with framework name as filename
df.to_csv('results/statistical_analysis/enhanced_rag.csv', index=False)

