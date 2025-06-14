from utils import trim_pdf
import pandas as pd
import os
import hashlib


data = pd.read_csv(
    r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\NEPAQuAD1_train.csv"
)

ROOT_PATH = r"D:\madhav\university\thesis\enhanced-rag\dataset\llm-for-environmental-review\pdfs_train"
EIS_filenames = ["eis_Goldrush_Mine_Project_FEIS_October_2023_508", "eis_Sea_Port_Oil_Terminal"]

def get_chunk_hash(chunk_text: str) -> str:
    return hashlib.md5(chunk_text.encode()).hexdigest()

# initialize metadata structure
trimmed_metadata = {
    "EIS_filename": [],
    "chunk_id": [],
    "token_count": [],
    "stats": [],
    "trimmed_pdf_path": [],
    "chunk_hash": [],
    "is_duplicate": []
}

processed_chunks = {}

for EIS_filename in EIS_filenames:
    print("\n" + "="*80)
    print(f"📄 Processing file: {EIS_filename}")
    print("="*80 + "\n")
    
    test_df = data[data['EIS_filename'] == EIS_filename]
    pdf_path = os.path.join(ROOT_PATH, EIS_filename + ".pdf")
    
    for index in range(len(test_df)):
        chunk_text = test_df["context"].iloc[index]
        chunk_id = test_df["ID"].iloc[index]
        chunk_hash = get_chunk_hash(chunk_text)
        
        chunk_key = (EIS_filename, chunk_hash)
        if chunk_key in processed_chunks:
            trimmed_pdf_path = processed_chunks[chunk_key]
            print(f"🔄 [DUPLICATE] Chunk {chunk_id} - Using existing trimmed PDF")
            trimmed_metadata["EIS_filename"].append(EIS_filename)
            trimmed_metadata["chunk_id"].append(chunk_id)
            trimmed_metadata["token_count"].append(None)
            trimmed_metadata["stats"].append("duplicate")
            trimmed_metadata["trimmed_pdf_path"].append(trimmed_pdf_path)
            trimmed_metadata["chunk_hash"].append(chunk_hash)
            trimmed_metadata["is_duplicate"].append(True)
            continue
        
        print(f"\n📝 Processing new chunk {chunk_id}")
        print("-"*50)
        
        max_pages_before = 300
        max_pages_after = 300
        max_tokens = 1_000_000
        output_path = os.path.join(ROOT_PATH, "trimmed", f"{chunk_id}_{EIS_filename}_trimmed.pdf")
        case_sensitive = False
        model_name = "gpt-4.1-mini"

        output_path, token_count, stats = trim_pdf(
            pdf_path, chunk_text, max_pages_before, max_pages_after, 
            max_tokens, output_path, case_sensitive, model_name
        )
        
        processed_chunks[chunk_key] = output_path
        
        print(f"✅ Completed chunk {chunk_id} - Token count: {token_count}")
        print(f"📊 Stats: {stats}")
        print("-"*50 + "\n")
        
        trimmed_metadata["EIS_filename"].append(EIS_filename)
        trimmed_metadata["chunk_id"].append(chunk_id)
        trimmed_metadata["token_count"].append(token_count)
        trimmed_metadata["stats"].append(stats)
        trimmed_metadata["trimmed_pdf_path"].append(output_path)
        trimmed_metadata["chunk_hash"].append(chunk_hash)
        trimmed_metadata["is_duplicate"].append(False)
# save metadata
metadata_df = pd.DataFrame(trimmed_metadata)
metadata_df.to_csv(os.path.join(ROOT_PATH, "trimmed_metadata.csv"), index=False)

# print summary
total_chunks = len(metadata_df)
unique_chunks = len(metadata_df[~metadata_df["is_duplicate"]])
duplicate_chunks = total_chunks - unique_chunks

print("\n" + "="*80)
print("📊 Processing Summary:")
print("="*80)
print(f"Total chunks processed: {total_chunks}")
print(f"Unique chunks: {unique_chunks}")
print(f"Duplicate chunks: {duplicate_chunks}")
print(f"Saved {duplicate_chunks} PDF generations by reusing existing trimmed PDFs")
print("="*80 + "\n")
