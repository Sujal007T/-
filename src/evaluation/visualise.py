import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Any

# Determine the project root assuming this script is in src/evaluation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def generate_architecture_diagram(save_path: str = None):
    """
    Generates a system architecture diagram for the disaster damage assessment pipeline.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "architecture_diagram.png")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 150)
    ax.axis('off')
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define styles
    box_styles = {
        'input_output': {'facecolor': '#e1f5fe', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
        'preprocessing': {'facecolor': '#fff9c4', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
        'track_a': {'facecolor': '#c8e6c9', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
        'ensemble': {'facecolor': '#ffe0b2', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
        'track_b': {'facecolor': '#e1bee7', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
        'evaluation': {'facecolor': '#f5f5f5', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.8'},
    }
    
    # Helper to draw a box
    def draw_box(ax, x, y, width, height, text, style, fontsize=10):
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, 
                                 edgecolor=style['edgecolor'], facecolor=style['facecolor'], 
                                 zorder=2)
        ax.add_patch(rect)
        # Use sub-labels font size 8 logic if sub-labels are indicated by newline
        parts = text.split('\n')
        if len(parts) > 1 and "Baseline" not in parts[0] and "Qwen" not in parts[0] and "QLoRA" not in parts[0]:
            # For multiline text with main title and subtitle
            ax.text(x + width/2, y + height/2 + 1.5, parts[0], ha='center', va='center', 
                    fontsize=fontsize, zorder=3, fontweight='bold')
            ax.text(x + width/2, y + height/2 - 1.5, '\n'.join(parts[1:]), ha='center', va='center', 
                    fontsize=8, zorder=3)
        else:
            # Default text
            ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                    fontsize=fontsize, zorder=3)
        return (x + width/2, y), (x + width/2, y + height)
    
    # Helper for arrows
    def draw_arrow(ax, start_point, end_point, zorder=1):
        ax.annotate('', xy=end_point, xycoords='data', xytext=start_point, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=0, shrinkB=0),
                    zorder=zorder)

    # 1. INPUT (y = 135)
    draw_box(ax, 10, 135, 80, 10, "[INPUT]\nPre-Disaster Satellite Image + Post-Disaster Satellite Image", box_styles['input_output'])
    
    # 2. PREPROCESSING (y = 115)
    draw_box(ax, 15, 115, 70, 10, "[PREPROCESSING]\nLazy Patch Cropper (α=0.8 expansion around building bbox)", box_styles['preprocessing'])
    
    # 3. TRACK A (y = 90)
    ax.text(50, 108, "[TRACK A — xBD]", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    draw_box(ax, 5, 90, 26, 12, "[Baseline A:\nResNet50 CNN]", box_styles['track_a'])
    draw_box(ax, 37, 90, 26, 12, "[Baseline B:\nChange-Detection]", box_styles['track_a'])
    draw_box(ax, 69, 90, 26, 12, "[Baseline C:\nBLIP-2 VLM Zero-Shot]", box_styles['track_a'])
    
    # 4. ENSEMBLE (y = 65)
    draw_box(ax, 20, 65, 60, 10, "[ENSEMBLE]\nWeighted Soft Voting (weights ∝ validation accuracy)", box_styles['ensemble'])
    
    # 5. TRACK B (y = 40)
    ax.text(50, 58, "[TRACK B — DisasterM3]", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    draw_box(ax, 15, 40, 30, 10, "[Qwen2.5-VL-3B\nZero-Shot]", box_styles['track_b'])
    draw_box(ax, 55, 40, 30, 10, "[QLoRA Fine-Tuned\n(r=16, α=32)]", box_styles['track_b'])
    
    # 6. OUTPUT (y = 20)
    draw_box(ax, 20, 20, 60, 10, "[OUTPUT]\nDamage Class Prediction + 3-Part Causal Report", box_styles['input_output'])
    
    # 7. EVALUATION (y = 0)
    draw_box(ax, 10, 0, 80, 10, "[EVALUATION]\nAccuracy | Macro-F1 | Per-Class F1 | Variance Score", box_styles['evaluation'])
    
    # Arrows
    draw_arrow(ax, (50, 135), (50, 125)) # Input to Preprocessing
    draw_arrow(ax, (50, 115), (50, 109)) # Preprocessing to Track A Text
    
    # Track A text to Track A models
    draw_arrow(ax, (50, 107), (18, 102))
    draw_arrow(ax, (50, 107), (50, 102))
    draw_arrow(ax, (50, 107), (82, 102))
    
    # Track A models to Ensemble
    draw_arrow(ax, (18, 90), (50, 75))
    draw_arrow(ax, (50, 90), (50, 75))
    draw_arrow(ax, (82, 90), (50, 75))
    
    # Ensemble to Track B Text
    draw_arrow(ax, (50, 65), (50, 59))
    
    # Track B Text to Qwen Zero-Shot
    draw_arrow(ax, (50, 57), (30, 50))
    
    # Qwen Zero-Shot to QLoRA
    draw_arrow(ax, (45, 45), (55, 45))
    
    # QLoRA to Output
    draw_arrow(ax, (70, 40), (50, 30))
    
    # Output to Evaluation
    draw_arrow(ax, (50, 20), (50, 10))

    plt.title("VLM-Based Disaster Damage Assessment — System Architecture", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_experimental_design_table(save_path: str = None):
    """
    Generates a table figure showing the 4-condition experimental design.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "experimental_design_table.png")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    col_labels = ['Condition', 'Label', 'Fine-tuned', 'Task Tokens', 'Patch Crops', 'Purpose']
    table_data = [
        ['A', 'Zero-shot baseline', '✗', '✗', '✗', 'Absolute baseline'],
        ['B', 'Zero-shot + prompting', '✗', '✓', '✗', 'Isolates prompt engineering'],
        ['C', 'Zero-shot + patches', '✗', '✓', '✓', 'Isolates preprocessing'],
        ['D', 'Fine-tuned (proposed)', '✓', '✓', '✓', 'Full proposed method']
    ]
    
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold')
        
    # Data rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == 4: # Condition D row
                cell.set_facecolor('#c8e6c9') # light green
            elif i % 2 != 0:
                cell.set_facecolor('white')
            else:
                cell.set_facecolor('#f5f5f5') # very light grey
                
    plt.title("Experimental Design — 4 Ablation Conditions", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_results_comparison(results_dict: Dict[str, Dict[str, float]], save_path: str):
    """
    Plots a grouped bar chart comparing Accuracy, Macro-F1, and Variance Score across 4 conditions.
    
    Args:
        results_dict: A dictionary mapping condition names to dictionaries of metric values.
            Example: {'Condition A': {'Accuracy': 0.7, 'Macro-F1': 0.65, 'Variance Score': 0.1}}
        save_path: Path to save the generated plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    conditions = list(results_dict.keys())
    metrics = ['Accuracy', 'Macro-F1', 'Variance Score']
    
    x = np.arange(len(conditions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[cond].get(metric, 0) for cond in conditions]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=metric)
        
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics across Experimental Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Ensure results dir exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    generate_architecture_diagram()
    generate_experimental_design_table()
    print("Figures saved to results/")
