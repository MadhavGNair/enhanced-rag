import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
with open('results/statistical_analysis/statistical_analysis.json', 'r') as f:
    data = json.load(f)

# Set up the plotting style for research papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for publication quality
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
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('RAG Framework Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

# Define framework categories and styling
framework_categories = {
    'baselines': {
        'frameworks': ['vanilla_rag', 'long_context'],
        'color': '#7f7f7f',  # Gray for baselines
        'alpha': 0.7,
        'hatch': None
    },
    'novel': {
        'frameworks': ['enhanced_rag'],
        'color': '#d62728',  # Red for novel framework (emphasis)
        'alpha': 0.9,
        'hatch': None
    },
    'existing': {
        'frameworks': ['self_route', 'hyde'],
        'color': '#1f77b4',  # Blue for existing methods
        'alpha': 0.8,
        'hatch': '//'
    }
}

# Metrics to plot
metrics = ['answer_correctness', 'context_recall', 'faithfulness']
metric_titles = ['Answer Correctness', 'Context Recall', 'Faithfulness']
metric_descriptions = [
    'Higher values indicate more accurate answers',
    'Higher values indicate better context retrieval',
    'Higher values indicate better generation from the retrieved context'
]

for idx, (metric, title, description) in enumerate(zip(metrics, metric_titles, metric_descriptions)):
    ax = axes[idx]
    
    # Collect data for this metric
    plot_data = []
    
    for category, config in framework_categories.items():
        for framework in config['frameworks']:
            # Skip long_context for context_recall and faithfulness
            if framework == 'long_context' and metric in ['context_recall', 'faithfulness']:
                continue
                
            evaluation = data['descriptive_statistics'][framework]
            
            # Check if metric exists and has valid data
            if metric in evaluation and evaluation[metric]['mean'] is not None:
                plot_data.append({
                    'framework': framework.replace('_', ' ').title(),
                    'mean': evaluation[metric]['mean'],
                    'std': evaluation[metric]['std'],
                    'median': evaluation[metric]['median'],
                    'category': category,
                    'color': config['color'],
                    'alpha': config['alpha'],
                    'hatch': config['hatch']
                })
    
    # Sort data to ensure consistent ordering: baselines, novel, existing
    category_order = ['novel', 'baselines', 'existing']
    plot_data.sort(key=lambda x: (category_order.index(x['category']), x['framework']))
    
    # Create bar plot
    x_pos = np.arange(len(plot_data))
    
    # Plot bars with different styling for each category
    bars = []
    for i, item in enumerate(plot_data):
        # Ensure error bars don't exceed plot boundaries
        error_val = min(item['std'], item['mean'] * 0.8)  # Cap error bars
        
        bar = ax.bar(i, item['mean'], yerr=error_val, capsize=5,
                    color=item['color'], alpha=item['alpha'], 
                    edgecolor='black', linewidth=0.8,
                    hatch=item['hatch'], error_kw={'linewidth': 1.5})
        bars.extend(bar)
        
        # Add value labels on bars
        ax.text(i, item['mean'] + error_val + 0.02,
                f'{item["mean"]:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Customize the plot
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    framework_labels = [item['framework'] for item in plot_data]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(framework_labels, rotation=45, ha='right', fontsize=10)
    
    # Set y-axis limits with some padding for error bars and labels
    max_val = max([item['mean'] + item['std'] for item in plot_data])
    # ax.set_ylim(0, min(1.0, max_val * 1.25))
    ax.set_ylim(0, 1.2)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add metric description as subtitle
    ax.text(0.5, -0.25, description, transform=ax.transAxes, 
            ha='center', va='top', fontsize=9, style='italic', color='gray')

# Create a comprehensive legend
legend_elements = []
for category, config in framework_categories.items():
    if category == 'baselines':
        label = 'Baseline Methods'
    elif category == 'novel':
        label = 'Novel Framework (Enhanced RAG)'
    else:
        label = 'Existing Advanced Methods'
    
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=config['color'], 
                                       alpha=config['alpha'], hatch=config['hatch'],
                                       edgecolor='black', linewidth=0.8, label=label))

# Place legend outside the plot area
fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
          ncol=3, frameon=True, fancybox=True, shadow=True)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.92)

# Save with high quality
plt.savefig('results/statistical_analysis/framework_comparison_plots.png', 
           dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('results/statistical_analysis/framework_comparison_plots.pdf', 
           bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

# # Print summary statistics for reference
# print("\nFramework Performance Summary:")
# print("=" * 50)
# for metric in metrics:
#     print(f"\n{metric.replace('_', ' ').title()}:")
#     print("-" * 30)
    
#     for category, config in framework_categories.items():
#         for framework in config['frameworks']:
#             if framework == 'long_context' and metric in ['context_recall', 'faithfulness']:
#                 continue
                
#             evaluation = data['descriptive_statistics'][framework]
#             if metric in evaluation and evaluation[metric]['mean'] is not None:
#                 mean_val = evaluation[metric]['mean']
#                 std_val = evaluation[metric]['std']
#                 print(f"{framework.replace('_', ' ').title():15}: {mean_val:.3f} ± {std_val:.3f}")