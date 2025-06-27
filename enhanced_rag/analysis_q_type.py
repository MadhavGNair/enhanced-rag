import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.stats import f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style (from plotting.py)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Frameworks and metrics
FRAMEWORKS = ['enhanced_rag', 'vanilla_rag', 'long_context', 'self_route', 'hyde']
METRICS = ['answer_correctness', 'context_recall', 'faithfulness']
METRIC_TITLES = ['Answer Correctness', 'Context Recall', 'Faithfulness']
METRIC_DESCRIPTIONS = [
    'Higher values indicate more accurate answers',
    'Higher values indicate better context retrieval',
    'Higher values indicate better generation from the retrieved context'
]

# Framework categories for plotting
FRAMEWORK_CATEGORIES = {
    'baselines': {
        'frameworks': ['vanilla_rag', 'long_context'],
        'color': '#7f7f7f',
        'alpha': 0.7,
        'hatch': None
    },
    'novel': {
        'frameworks': ['enhanced_rag'],
        'color': '#d62728',
        'alpha': 0.9,
        'hatch': None
    },
    'existing': {
        'frameworks': ['self_route', 'hyde'],
        'color': '#1f77b4',
        'alpha': 0.8,
        'hatch': '//'
    }
}

# Output directory
OUTPUT_DIR = Path("results/statistical_analysis/per_q_type")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
RAW_DATA_PATH = Path("results/statistical_analysis/raw_data.csv")
df = pd.read_csv(RAW_DATA_PATH)

question_types = sorted(df['question_type'].dropna().unique())

all_results = {}

def convert_to_json_serializable(obj):
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

for q_type in question_types:
    qtype_results = {}
    qtype_df = df[df['question_type'] == q_type]
    
    # Descriptive stats per framework
    desc_stats = {}
    for fw in FRAMEWORKS:
        fw_df = qtype_df[qtype_df['framework'] == fw]
        desc_stats[fw] = {}
        for metric in METRICS:
            vals = fw_df[metric].dropna()
            desc_stats[fw][metric] = {
                'count': len(vals),
                'mean': float(vals.mean()) if len(vals) > 0 else None,
                'std': float(vals.std()) if len(vals) > 0 else None,
                'median': float(vals.median()) if len(vals) > 0 else None,
                'min': float(vals.min()) if len(vals) > 0 else None,
                'max': float(vals.max()) if len(vals) > 0 else None
            }
    qtype_results['descriptive_statistics'] = desc_stats

    # Statistical tests per metric
    stat_tests = {}
    for metric in METRICS:
        fw_data = {fw: qtype_df[qtype_df['framework'] == fw][metric].dropna().tolist() for fw in FRAMEWORKS}
        # Only keep frameworks with data
        fw_data = {fw: vals for fw, vals in fw_data.items() if len(vals) > 0}
        test_result = {'frameworks_compared': list(fw_data.keys()), 'sample_sizes': {fw: len(vals) for fw, vals in fw_data.items()}}
        if len(fw_data) < 2:
            test_result['note'] = 'Insufficient frameworks with data for comparison'
            stat_tests[metric] = test_result
            continue
        groups = list(fw_data.values())
        # Normality
        normality = {}
        for fw, vals in fw_data.items():
            if len(vals) >= 3:
                try:
                    stat, p = stats.shapiro(vals)
                    normality[fw] = {'statistic': float(stat), 'p_value': float(p), 'is_normal': p > 0.05}
                except Exception as e:
                    normality[fw] = {'error': str(e)}
        test_result['normality_tests'] = normality
        # ANOVA
        try:
            f_stat, anova_p = f_oneway(*groups)
            test_result['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'significant': anova_p < 0.05
            }
        except Exception as e:
            test_result['anova'] = {'error': str(e)}
        # Kruskal-Wallis
        try:
            h_stat, kruskal_p = kruskal(*groups)
            test_result['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(kruskal_p),
                'significant': kruskal_p < 0.05
            }
        except Exception as e:
            test_result['kruskal_wallis'] = {'error': str(e)}
        # Pairwise Mann-Whitney if significant
        if (test_result.get('anova', {}).get('significant', False) or test_result.get('kruskal_wallis', {}).get('significant', False)):
            pairwise = {}
            fw_list = list(fw_data.keys())
            for i in range(len(fw_list)):
                for j in range(i+1, len(fw_list)):
                    fw1, fw2 = fw_list[i], fw_list[j]
                    try:
                        u_stat, u_p = stats.mannwhitneyu(fw_data[fw1], fw_data[fw2], alternative='two-sided')
                        pairwise[f'{fw1}_vs_{fw2}'] = {
                            'u_statistic': float(u_stat),
                            'p_value': float(u_p),
                            'significant': u_p < 0.05
                        }
                    except Exception as e:
                        pairwise[f'{fw1}_vs_{fw2}'] = {'error': str(e)}
            test_result['pairwise_comparisons'] = pairwise
        stat_tests[metric] = test_result
    qtype_results['statistical_tests'] = stat_tests

    # Save results JSON
    with open(OUTPUT_DIR / f'{q_type}_analysis.json', 'w') as f:
        json.dump(convert_to_json_serializable(qtype_results), f, indent=2)
    all_results[q_type] = qtype_results

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Framework Comparison for Question Type: {q_type.title()}', fontsize=16, fontweight='bold', y=0.98)
    for idx, (metric, title, description) in enumerate(zip(METRICS, METRIC_TITLES, METRIC_DESCRIPTIONS)):
        ax = axes[idx]
        plot_data = []
        for category, config in FRAMEWORK_CATEGORIES.items():
            for fw in config['frameworks']:
                stats_ = desc_stats[fw][metric]
                if stats_['mean'] is not None:
                    plot_data.append({
                        'framework': fw.replace('_', ' ').title(),
                        'mean': stats_['mean'],
                        'std': stats_['std'],
                        'category': category,
                        'color': config['color'],
                        'alpha': config['alpha'],
                        'hatch': config['hatch']
                    })
        # Sort for consistent order
        category_order = ['novel', 'baselines', 'existing']
        plot_data.sort(key=lambda x: (category_order.index(x['category']), x['framework']))
        x_pos = np.arange(len(plot_data))
        bars = []
        for i, item in enumerate(plot_data):
            error_val = min(item['std'], item['mean'] * 0.8) if item['std'] is not None and item['mean'] is not None else 0
            bar = ax.bar(i, item['mean'], yerr=error_val, capsize=5,
                        color=item['color'], alpha=item['alpha'], 
                        edgecolor='black', linewidth=0.8,
                        hatch=item['hatch'], error_kw={'linewidth': 1.5})
            bars.extend(bar)
            ax.text(i, item['mean'] + (error_val if error_val else 0) + 0.02,
                    f'{item["mean"]:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        framework_labels = [item['framework'] for item in plot_data]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(framework_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.25, description, transform=ax.transAxes, 
                ha='center', va='top', fontsize=9, style='italic', color='gray')
    # Legend
    legend_elements = []
    for category, config in FRAMEWORK_CATEGORIES.items():
        if category == 'baselines':
            label = 'Baseline Methods'
        elif category == 'novel':
            label = 'Novel Framework (Enhanced RAG)'
        else:
            label = 'Existing Advanced Methods'
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=config['color'], 
                                           alpha=config['alpha'], hatch=config['hatch'],
                                           edgecolor='black', linewidth=0.8, label=label))
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=3, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)
    plt.show()

# Save all results summary
with open(OUTPUT_DIR / 'all_qtype_analysis.json', 'w') as f:
    json.dump(convert_to_json_serializable(all_results), f, indent=2)

print(f"Analysis complete. Results and figures saved in {OUTPUT_DIR}")
