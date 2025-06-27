import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

def load_framework_data():
    """Load data from all framework JSON files"""
    
    # Define the path to the gpt_evaluation directory
    data_dir = Path("results/gpt_evaluation")
    
    # Framework names
    frameworks = ['enhanced_rag', 'vanilla_rag', 'long_context', 'self_route', 'hyde']
    
    all_data = []
    
    for framework in frameworks:
        file_path = data_dir / f"{framework}.json"
        
        print(f"Loading {framework} data...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            # Base row data
            row = {
                'framework': framework,
                'index': item['index'],
                'ID': item['ID'],
                'question_type': item['question_type'],
                'answer_correctness': item['evaluation']['answer_correctness'],
                'context_recall': item['evaluation']['context_recall'],
                'faithfulness': item['evaluation']['faithfulness']
            }
            
            # Handle self_route special case
            if framework == 'self_route':
                row['response_type'] = item['response_type']
            else:
                row['response_type'] = None
                
            all_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"Loaded {len(df)} records from {len(frameworks)} frameworks")
    print(f"Records per framework: {len(df) // len(frameworks)}")
    
    return df

def save_raw_data(df):
    """Save raw data to CSV"""
    
    # Create directory if it doesn't exist
    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / "raw_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Raw data saved to {output_path}")

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for each framework"""
    
    metrics = ['answer_correctness', 'context_recall', 'faithfulness']
    frameworks = df['framework'].unique()
    
    stats_results = {}
    
    for framework in frameworks:
        framework_data = df[df['framework'] == framework]
        stats_results[framework] = {}
        
        for metric in metrics:
            # Get non-null values for the metric
            values = framework_data[metric].dropna()
            
            if len(values) > 0:
                stats_results[framework][metric] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'null_count': int(framework_data[metric].isnull().sum())
                }
            else:
                stats_results[framework][metric] = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'median': None,
                    'min': None,
                    'max': None,
                    'q25': None,
                    'q75': None,
                    'null_count': int(len(framework_data))
                }
    
    return stats_results

def calculate_question_type_stats(df):
    """Calculate statistics by question type"""
    
    metrics = ['answer_correctness', 'context_recall', 'faithfulness']
    question_types = df['question_type'].unique()
    frameworks = df['framework'].unique()
    
    question_type_stats = {}
    
    for qt in question_types:
        question_type_stats[qt] = {}
        
        for framework in frameworks:
            subset = df[(df['framework'] == framework) & (df['question_type'] == qt)]
            question_type_stats[qt][framework] = {}
            
            for metric in metrics:
                values = subset[metric].dropna()
                
                if len(values) > 0:
                    question_type_stats[qt][framework][metric] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
                else:
                    question_type_stats[qt][framework][metric] = {
                        'count': 0,
                        'mean': None,
                        'std': None
                    }
    
    return question_type_stats

def calculate_self_route_breakdown(df):
    """Calculate breakdown for self_route by response_type"""
    
    self_route_data = df[df['framework'] == 'self_route']
    
    if len(self_route_data) == 0:
        return {}
    
    breakdown = {}
    response_types = self_route_data['response_type'].unique()
    metrics = ['answer_correctness', 'context_recall', 'faithfulness']
    
    for response_type in response_types:
        subset = self_route_data[self_route_data['response_type'] == response_type]
        breakdown[response_type] = {}
        
        for metric in metrics:
            values = subset[metric].dropna()
            
            if len(values) > 0:
                breakdown[response_type][metric] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median())
                }
            else:
                breakdown[response_type][metric] = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'median': None
                }
    
    # Add overall counts
    breakdown['total_counts'] = {
        'vanilla_rag_responses': int(len(self_route_data[self_route_data['response_type'] == 'vanilla_rag'])),
        'long_context_responses': int(len(self_route_data[self_route_data['response_type'] == 'long_context']))
    }
    
    return breakdown

def perform_statistical_tests(df):
    """Perform statistical tests comparing frameworks"""
    
    frameworks = df['framework'].unique()
    metrics = ['answer_correctness', 'context_recall', 'faithfulness']
    
    test_results = {}
    
    for metric in metrics:
        print(f"Performing tests for {metric}...")
        
        # Collect data for frameworks that have this metric
        framework_data = {}
        
        for framework in frameworks:
            values = df[df['framework'] == framework][metric].dropna()
            if len(values) > 0:
                framework_data[framework] = values.tolist()
        
        if len(framework_data) < 2:
            test_results[metric] = {
                'note': 'Insufficient frameworks with data for comparison'
            }
            continue
        
        # Prepare data for tests
        groups = list(framework_data.values())
        group_names = list(framework_data.keys())
        
        test_results[metric] = {
            'frameworks_compared': group_names,
            'sample_sizes': {name: len(data) for name, data in framework_data.items()}
        }
        
        # Normality tests (Shapiro-Wilk for each group)
        normality_results = {}
        for name, data in framework_data.items():
            if len(data) >= 3:  # Minimum sample size for Shapiro-Wilk
                try:
                    stat, p_val = stats.shapiro(data)
                    normality_results[name] = {
                        'statistic': float(stat),
                        'p_value': float(p_val),
                        'is_normal': p_val > 0.05
                    }
                except:
                    normality_results[name] = {'error': 'Could not perform normality test'}
        
        test_results[metric]['normality_tests'] = normality_results
        
        # Check if all groups are normally distributed
        all_normal = all(
            result.get('is_normal', False) 
            for result in normality_results.values() 
            if 'is_normal' in result
        )
        
        # ANOVA (parametric test)
        try:
            f_stat, anova_p = f_oneway(*groups)
            test_results[metric]['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'significant': anova_p < 0.05,
                'interpretation': 'Significant differences between frameworks' if anova_p < 0.05 else 'No significant differences between frameworks'
            }
        except Exception as e:
            test_results[metric]['anova'] = {'error': str(e)}
        
        # Kruskal-Wallis (non-parametric test)
        try:
            h_stat, kruskal_p = kruskal(*groups)
            test_results[metric]['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(kruskal_p),
                'significant': kruskal_p < 0.05,
                'interpretation': 'Significant differences between frameworks' if kruskal_p < 0.05 else 'No significant differences between frameworks'
            }
        except Exception as e:
            test_results[metric]['kruskal_wallis'] = {'error': str(e)}
        
        # Pairwise comparisons (if significant differences found)
        if (test_results[metric].get('anova', {}).get('significant', False) or 
            test_results[metric].get('kruskal_wallis', {}).get('significant', False)):
            
            pairwise_results = {}
            
            for i, (name1, data1) in enumerate(framework_data.items()):
                for j, (name2, data2) in enumerate(framework_data.items()):
                    if i < j:  # Avoid duplicate comparisons
                        pair_name = f"{name1}_vs_{name2}"
                        
                        # Mann-Whitney U test (non-parametric)
                        try:
                            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            pairwise_results[pair_name] = {
                                'u_statistic': float(u_stat),
                                'p_value': float(u_p),
                                'significant': u_p < 0.05,
                                'effect_size': float(u_stat / (len(data1) * len(data2)))  # Simple effect size
                            }
                        except Exception as e:
                            pairwise_results[pair_name] = {'error': str(e)}
            
            test_results[metric]['pairwise_comparisons'] = pairwise_results
    
    return test_results

def generate_summary_insights(descriptive_stats, test_results, self_route_breakdown):
    """Generate summary insights from the analysis"""
    
    insights = {
        'framework_ranking': {},
        'key_findings': [],
        'recommendations': []
    }
    
    # Rank frameworks by each metric
    for metric in ['answer_correctness', 'context_recall', 'faithfulness']:
        rankings = []
        
        for framework, stats in descriptive_stats.items():
            if stats[metric]['mean'] is not None:
                rankings.append({
                    'framework': framework,
                    'mean': stats[metric]['mean'],
                    'std': stats[metric]['std']
                })
        
        # Sort by mean score (descending)
        rankings.sort(key=lambda x: x['mean'], reverse=True)
        insights['framework_ranking'][metric] = rankings
    
    # Key findings
    if 'answer_correctness' in test_results:
        ac_test = test_results['answer_correctness']
        if ac_test.get('anova', {}).get('significant', False):
            insights['key_findings'].append('Significant differences found in answer correctness between frameworks')
        else:
            insights['key_findings'].append('No significant differences in answer correctness between frameworks')
    
    # Self-route analysis
    if self_route_breakdown:
        total_counts = self_route_breakdown.get('total_counts', {})
        vanilla_count = total_counts.get('vanilla_rag_responses', 0)
        long_context_count = total_counts.get('long_context_responses', 0)
        total = vanilla_count + long_context_count
        
        if total > 0:
            vanilla_pct = (vanilla_count / total) * 100
            insights['key_findings'].append(
                f'Self-route used vanilla_rag {vanilla_pct:.1f}% of the time and long_context {100-vanilla_pct:.1f}% of the time'
            )
    
    return insights

def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON serializable types"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return obj

def save_analysis_results(descriptive_stats, test_results, question_type_stats, 
                         self_route_breakdown, insights):
    """Save all analysis results to JSON"""
    
    # Create directory if it doesn't exist
    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile all results
    results = {
        'analysis_summary': {
            'description': 'Statistical analysis of RAG framework performance',
            'frameworks_analyzed': ['enhanced_rag', 'vanilla_rag', 'long_context', 'self_route', 'hyde'],
            'metrics_analyzed': ['answer_correctness', 'context_recall', 'faithfulness'],
            'total_samples_per_framework': 1079
        },
        'descriptive_statistics': descriptive_stats,
        'statistical_tests': test_results,
        'question_type_analysis': question_type_stats,
        'self_route_breakdown': self_route_breakdown,
        'insights_and_rankings': insights
    }
    
    # Convert to JSON serializable format
    results = convert_to_json_serializable(results)
    
    # Save to JSON
    output_path = output_dir / "statistical_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Statistical analysis results saved to {output_path}")

def main():
    """Main analysis function"""
    
    print("Starting comprehensive statistical analysis...")
    print("=" * 50)
    
    # Load data
    df = load_framework_data()
    
    # Save raw data
    save_raw_data(df)
    
    # Calculate descriptive statistics
    print("\nCalculating descriptive statistics...")
    descriptive_stats = calculate_descriptive_stats(df)
    
    # Calculate question type statistics
    print("Analyzing performance by question type...")
    question_type_stats = calculate_question_type_stats(df)
    
    # Self-route breakdown
    print("Analyzing self-route breakdown...")
    self_route_breakdown = calculate_self_route_breakdown(df)
    
    # Perform statistical tests
    print("Performing statistical significance tests...")
    test_results = perform_statistical_tests(df)
    
    # Generate insights
    print("Generating insights and rankings...")
    insights = generate_summary_insights(descriptive_stats, test_results, self_route_breakdown)
    
    # Save results
    save_analysis_results(descriptive_stats, test_results, question_type_stats, 
                         self_route_breakdown, insights)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print(f"- Total records analyzed: {len(df)}")
    print(f"- Frameworks: {', '.join(df['framework'].unique())}")
    print(f"- Question types: {', '.join(df['question_type'].unique())}")
    
    # Show top performer for answer correctness
    if 'answer_correctness' in insights['framework_ranking']:
        top_framework = insights['framework_ranking']['answer_correctness'][0]
        print(f"- Best answer correctness: {top_framework['framework']} (mean: {top_framework['mean']:.4f})")

if __name__ == "__main__":
    main()
