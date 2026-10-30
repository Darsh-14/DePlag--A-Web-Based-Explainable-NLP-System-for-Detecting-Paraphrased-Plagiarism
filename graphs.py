import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generate_speed_accuracy_plot():
    print("Generating Figure 1...")
    
    # Data from Table in Section 4.1
    data = {
        'Model': [
            'BERT-Base', 'sentence-t5-base', 'SimCSE-BERT', 'BGE-Small', 
            'Specter', 'GTR-T5', 'GTE-Base', 'RoBERTa-Base', 
            'MiniLM-L6', 'MiniLM-L12', 'MiniLM-L3'
        ],
        'Accuracy': [0.9176, 0.8818, 0.8375, 0.8164, 0.7662, 0.7647, 0.6684, 0.6530, 0.4866, 0.4282, 0.3510],
        'Latency_ms': [332.26, 79.20, 318.95, 94.20, 312.27, 63.11, 437.58, 315.22, 44.96, 91.85, 26.61],
        'Type': ['Heavy', 'Selected', 'Heavy', 'Light', 'Heavy', 'Light', 'Heavy', 'Heavy', 'Light', 'Light', 'Light']
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    
    
    sns.scatterplot(
        data=df, x='Latency_ms', y='Accuracy', 
        hue='Type', style='Type', s=150, 
        palette={'Heavy': '#95a5a6', 'Light': '#f1c40f', 'Selected': '#e74c3c'}
    )

    
    selected = df[df['Type'] == 'Selected'].iloc[0]
    plt.text(selected['Latency_ms']+10, selected['Accuracy'], 
             f"  {selected['Model']} (Ours)", fontweight='bold', color='#c0392b', va='center')

    plt.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='Real-Time Limit (100ms)')
    
    plt.title('Pareto Efficiency: Semantic Fidelity vs. Inference Latency', fontweight='bold')
    plt.xlabel('Inference Latency (ms/sentence)')
    plt.ylabel('Semantic Accuracy (Spearman $\\rho$)')
    plt.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    plt.savefig('fig1_speed.png', dpi=300)
    plt.close()

def generate_performance_bar_chart():
    """Generates Figure 2: F1 Score Comparison (MSRP vs PAWS)"""
    print("Generating Figure 2...")
    
    models = ['TF-IDF', 'Word2Vec', 'BERT (Cross)', 'Hybrid SBERT (Ours)']
    msrp_scores = [67, 72, 88, 86] # Simple
    paws_scores = [45, 52, 84, 81] # Adversarial

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, msrp_scores, width, label='MSRP (Standard)', color='#34495e')
    plt.bar(x + width/2, paws_scores, width, label='PAWS (Adversarial)', color='#e74c3c')

    # Labels
    plt.ylabel('F1-Score (%)', fontweight='bold')
    plt.title('Robustness Analysis: Detection Performance', fontweight='bold')
    plt.xticks(x, models)
    plt.ylim(0, 100)
    plt.legend()

    plt.tight_layout()
    plt.savefig('fig2_performance.png', dpi=300)
    plt.close()

def generate_confusion_matrix():
    """Figure 3: Confusion Matrix for PAWS"""
    print("Generating Figure 3...")
    
    cm_data = np.array([[842, 115], 
                        [130, 895]]) 

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_data, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Original', 'Plagiarized'], 
        yticklabels=['Original', 'Plagiarized'],
        annot_kws={"size": 14, "weight": "bold"},
        cbar=False
    )

    plt.title('Confusion Matrix (Adversarial Dataset)', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('fig3_confusion.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_speed_accuracy_plot()
    generate_performance_bar_chart()
    generate_confusion_matrix()
    print(" All graphs generated successfully!")