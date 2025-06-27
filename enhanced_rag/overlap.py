import json
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two text strings.
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    # Preprocess text: lowercase, remove punctuation, split into words
    def preprocess(text):
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and remove empty strings
        words = [word.strip() for word in text.split() if word.strip()]
        return set(words)
    
    set1 = preprocess(text1)
    set2 = preprocess(text2)
    
    if not set1 and not set2:
        return 1.0  # Both empty sets are considered identical
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def analyze_retrieved_contexts(json_path: str) -> Dict:
    """
    Analyze the diversity of retrieved contexts based on page numbers and content overlap.
    
    Args:
        json_path: Path to the JSON file containing retrieved contexts
        
    Returns:
        Dictionary with analysis results
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize result structure
    analysis = {
        'page_number_analysis': {
            'three_different_pages': {'count': 0, 'ids': []},
            'two_different_pages': {'count': 0, 'ids': []},
            'same_pages': {'count': 0, 'ids': []}
        },
        'content_overlap_analysis': {
            'high_overlap_70_plus': {'count': 0, 'ids': []},
            'medium_overlap_50_70': {'count': 0, 'ids': []},
            'low_overlap_below_50': {'count': 0, 'ids': []}
        },
        'detailed_analysis': {}
    }
    
    for item in data:
        if 'ID' not in item or 'retrieved_contexts' not in item:
            continue
            
        item_id = item['ID']
        contexts = item['retrieved_contexts']
        
        if len(contexts) != 3:
            continue  # Skip if not exactly 3 contexts
        
        # Extract page numbers and content
        page_numbers = [ctx.get('page_number', -1) for ctx in contexts]
        page_contents = [ctx.get('page_content', '') for ctx in contexts]
        
        # Analyze page number diversity
        unique_pages = len(set(page_numbers))
        
        if unique_pages == 3:
            analysis['page_number_analysis']['three_different_pages']['count'] += 1
            analysis['page_number_analysis']['three_different_pages']['ids'].append(item_id)
        elif unique_pages == 2:
            analysis['page_number_analysis']['two_different_pages']['count'] += 1
            analysis['page_number_analysis']['two_different_pages']['ids'].append(item_id)
        else:  # unique_pages == 1
            analysis['page_number_analysis']['same_pages']['count'] += 1
            analysis['page_number_analysis']['same_pages']['ids'].append(item_id)
        
        # Analyze content overlap
        overlap_scores = []
        for i in range(len(page_contents)):
            for j in range(i + 1, len(page_contents)):
                similarity = compute_jaccard_similarity(page_contents[i], page_contents[j])
                overlap_scores.append(similarity)
        
        # Determine overlap category based on average similarity
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        
        if avg_overlap >= 0.7:
            analysis['content_overlap_analysis']['high_overlap_70_plus']['count'] += 1
            analysis['content_overlap_analysis']['high_overlap_70_plus']['ids'].append(item_id)
        elif avg_overlap >= 0.5:
            analysis['content_overlap_analysis']['medium_overlap_50_70']['count'] += 1
            analysis['content_overlap_analysis']['medium_overlap_50_70']['ids'].append(item_id)
        else:
            analysis['content_overlap_analysis']['low_overlap_below_50']['count'] += 1
            analysis['content_overlap_analysis']['low_overlap_below_50']['ids'].append(item_id)
        
        # Store detailed analysis for this item
        analysis['detailed_analysis'][item_id] = {
            'page_numbers': page_numbers,
            'unique_pages': unique_pages,
            'overlap_scores': overlap_scores,
            'average_overlap': avg_overlap
        }
    
    return analysis

def print_analysis_summary(analysis: Dict):
    """
    Print a formatted summary of the analysis results.
    
    Args:
        analysis: Analysis results dictionary
    """
    print("=" * 80)
    print("RETRIEVED CONTEXTS DIVERSITY ANALYSIS")
    print("=" * 80)
    
    print("\n1. PAGE NUMBER DIVERSITY ANALYSIS:")
    print("-" * 40)
    page_analysis = analysis['page_number_analysis']
    print(f"• IDs with 3 different page numbers: {page_analysis['three_different_pages']['count']}")
    print(f"• IDs with 2 different page numbers: {page_analysis['two_different_pages']['count']}")
    print(f"• IDs with same page numbers: {page_analysis['same_pages']['count']}")
    
    print("\n2. CONTENT OVERLAP ANALYSIS:")
    print("-" * 40)
    overlap_analysis = analysis['content_overlap_analysis']
    print(f"• IDs with high overlap (≥70%): {overlap_analysis['high_overlap_70_plus']['count']}")
    print(f"• IDs with medium overlap (50-70%): {overlap_analysis['medium_overlap_50_70']['count']}")
    print(f"• IDs with low overlap (<50%): {overlap_analysis['low_overlap_below_50']['count']}")
    
    print("\n3. DETAILED BREAKDOWN:")
    print("-" * 40)
    
    # Show some example IDs for each category
    print("\nPage Number Diversity Examples:")
    if page_analysis['three_different_pages']['ids']:
        print(f"  3 different pages (first 5): {page_analysis['three_different_pages']['ids'][:5]}")
    if page_analysis['two_different_pages']['ids']:
        print(f"  2 different pages (first 5): {page_analysis['two_different_pages']['ids'][:5]}")
    if page_analysis['same_pages']['ids']:
        print(f"  Same pages (first 5): {page_analysis['same_pages']['ids'][:5]}")
    
    print("\nContent Overlap Examples:")
    if overlap_analysis['high_overlap_70_plus']['ids']:
        print(f"  High overlap (first 5): {overlap_analysis['high_overlap_70_plus']['ids'][:5]}")
    if overlap_analysis['medium_overlap_50_70']['ids']:
        print(f"  Medium overlap (first 5): {overlap_analysis['medium_overlap_50_70']['ids'][:5]}")
    if overlap_analysis['low_overlap_below_50']['ids']:
        print(f"  Low overlap (first 5): {overlap_analysis['low_overlap_below_50']['ids'][:5]}")

def main():
    """Main function to run the analysis."""
    JSON_PATH = 'results/raw_results/enhanced_rag.json'
    
    try:
        print("Loading and analyzing enhanced_rag.json...")
        analysis = analyze_retrieved_contexts(JSON_PATH)
        
        # Print summary
        print_analysis_summary(analysis)
        
        # Save detailed results to file
        output_path = 'results/context_diversity_analysis.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed analysis saved to: {output_path}")
        
        # Print some statistics
        total_items = len(analysis['detailed_analysis'])
        print(f"\nTotal items analyzed: {total_items}")
        
        # Calculate percentages
        page_analysis = analysis['page_number_analysis']
        overlap_analysis = analysis['content_overlap_analysis']
        
        print(f"\nPage Number Diversity Percentages:")
        print(f"  3 different pages: {page_analysis['three_different_pages']['count']/total_items*100:.1f}%")
        print(f"  2 different pages: {page_analysis['two_different_pages']['count']/total_items*100:.1f}%")
        print(f"  Same pages: {page_analysis['same_pages']['count']/total_items*100:.1f}%")
        
        print(f"\nContent Overlap Percentages:")
        print(f"  High overlap (≥70%): {overlap_analysis['high_overlap_70_plus']['count']/total_items*100:.1f}%")
        print(f"  Medium overlap (50-70%): {overlap_analysis['medium_overlap_50_70']['count']/total_items*100:.1f}%")
        print(f"  Low overlap (<50%): {overlap_analysis['low_overlap_below_50']['count']/total_items*100:.1f}%")
        
    except FileNotFoundError:
        print(f"Error: File {JSON_PATH} not found!")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {JSON_PATH}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 