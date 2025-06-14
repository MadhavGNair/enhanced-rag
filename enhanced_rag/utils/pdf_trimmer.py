import os
import random
import re
from typing import Optional, Tuple
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm
import difflib
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Using approximate token counting.")
    print("Install with: pip install tiktoken")


def trim_pdf(
    pdf_path: str, 
    chunk_text: str, 
    max_pages_before: int, 
    max_pages_after: int, 
    max_tokens: int = 1_000_000,
    output_path: Optional[str] = None,
    case_sensitive: bool = False,
    model_name: str = "gpt-4.1-mini"
) -> Tuple[str, int, dict]:    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    chunk_normalized = normalize_text(chunk_text, case_sensitive)
    
    chunk_page = find_chunk_page(pdf_path, chunk_normalized, case_sensitive)
    
    if chunk_page is None:
        raise ValueError(f"Chunk not found in PDF: {chunk_text[:50]}...")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    page_range, token_count, stats = find_optimal_page_range(
        pdf_path, chunk_page, total_pages, max_pages_before, max_pages_after, 
        max_tokens, model_name
    )
    
    start_page, end_page = page_range
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"{base_name}_chunk_p{chunk_page+1}_pages_{start_page+1}-{end_page+1}_{token_count}tokens.pdf"
    
    create_pdf_subset(pdf_path, start_page, end_page, output_path)
    
    print(f"Chunk found on page {chunk_page + 1}")
    print(f"Extracted pages {start_page + 1} to {end_page + 1} ({end_page - start_page + 1} pages)")
    print(f"Final token count: {token_count:,}")
    print(f"Chunk position: {stats['chunk_position']}")
    print(f"Pages before chunk: {chunk_page - start_page}")
    print(f"Pages after chunk: {end_page - chunk_page}")
    print(f"Output saved to: {output_path}")
    
    return output_path, token_count, stats


def normalize_text(text: str, case_sensitive: bool = False) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    
    if not case_sensitive:
        text = text.lower()
    
    return text


def find_chunk_page(pdf_path: str, chunk_normalized: str, case_sensitive: bool) -> Optional[int]:
    print("Searching for chunk in PDF...")
    best_match = None
    best_ratio = 0.0
    threshold = 0.60
    
    # Split chunk into sentences for better matching
    chunk_sentences = [s.strip() for s in chunk_normalized.split('.') if s.strip()]
    if not chunk_sentences:
        chunk_sentences = [chunk_normalized]  # If no periods, use whole chunk
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in tqdm(range(len(pdf.pages)), desc="Scanning pages"):
            try:
                page_text = pdf.pages[page_num].extract_text()
                
                if page_text:
                    page_text_normalized = normalize_text(page_text, case_sensitive)
                    
                    # First try exact substring match
                    if chunk_normalized in page_text_normalized:
                        print(f"Found exact match on page {page_num + 1}")
                        return page_num
                    
                    # If no exact match, try to find best matching substring
                    # Look for matches of individual sentences
                    matching_sentences = 0
                    for sentence in chunk_sentences:
                        if sentence in page_text_normalized:
                            matching_sentences += 1
                    
                    # Calculate ratio based on how many sentences matched
                    ratio = matching_sentences / len(chunk_sentences)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = page_num
                        print(f"New best match found on page {page_num + 1} with ratio: {ratio:.2f} ({matching_sentences}/{len(chunk_sentences)} sentences matched)")
                        
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                continue
    
    if best_match is not None and best_ratio >= threshold:
        print(f"Found best matching chunk on page {best_match + 1} with similarity ratio: {best_ratio:.2f}")
        return best_match
    
    print(f"No match found. Best similarity ratio was: {best_ratio:.2f}")
    return None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    else:
        return len(text) // 4


def extract_text_from_page_range(pdf_path: str, start_page: int, end_page: int) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page + 1):
            if page_num < len(pdf.pages):
                try:
                    page_text = pdf.pages[page_num].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
    
    return "\n".join(text_parts)


def find_optimal_page_range(
    pdf_path: str, 
    chunk_page: int, 
    total_pages: int,
    max_pages_before: int, 
    max_pages_after: int, 
    max_tokens: int,
    model_name: str
) -> Tuple[Tuple[int, int], int, dict]:
    print("Finding optimal page range...")
    candidates = []
    attempts = 10  # Reduced from 20 to 10
    
    for _ in tqdm(range(attempts), desc="Trying page combinations"):
        pages_before = random.randint(0, min(max_pages_before, chunk_page))
        pages_after = random.randint(0, min(max_pages_after, total_pages - chunk_page - 1))
        
        start_page = chunk_page - pages_before
        end_page = chunk_page + pages_after
        
        text = extract_text_from_page_range(pdf_path, start_page, end_page)
        token_count = count_tokens(text, model_name)
        
        if token_count <= max_tokens:
            total_pages_in_range = end_page - start_page + 1
            chunk_position = (chunk_page - start_page) / total_pages_in_range if total_pages_in_range > 1 else 0.5
            
            candidates.append({
                'range': (start_page, end_page),
                'token_count': token_count,
                'total_pages': total_pages_in_range,
                'pages_before': pages_before,
                'pages_after': pages_after,
                'chunk_position': chunk_position,
                'text': text
            })
    
    if not candidates:
        print("No valid combinations found, trying single page...")
        text = extract_text_from_page_range(pdf_path, chunk_page, chunk_page)
        token_count = count_tokens(text, model_name)
        
        if token_count <= max_tokens:
            return ((chunk_page, chunk_page), token_count, {
                'chunk_position': 0.5,
                'total_attempts': attempts,
                'successful_candidates': 1
            })
        else:
            raise ValueError(f"Even single page with chunk exceeds token limit: {token_count:,} tokens")
    
    candidates.sort(key=lambda x: x['total_pages'], reverse=True)
    
    top_candidates = candidates[:min(5, len(candidates))]
    selected = random.choice(top_candidates)
    
    stats = {
        'chunk_position': f"{selected['chunk_position']:.1%}",
        'total_attempts': attempts,
        'successful_candidates': len(candidates),
        'pages_before': selected['pages_before'],
        'pages_after': selected['pages_after']
    }
    
    return selected['range'], selected['token_count'], stats


def create_pdf_subset(pdf_path: str, start_page: int, end_page: int, output_path: str):
    print(f"Creating trimmed PDF with pages {start_page + 1} to {end_page + 1}...")
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    for page_num in tqdm(range(start_page, end_page + 1), desc="Copying pages"):
        if page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])
    
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)
    print(f"Trimmed PDF saved to: {output_path}")