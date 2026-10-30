import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#E6F3FF', edgecolor='black', shape='rect'):
    """Helper to draw a box with text"""
    if shape == 'rect':
        rect = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=edgecolor, facecolor=color, zorder=2
        )
        ax.add_patch(rect)
    elif shape == 'diamond':
        rect = patches.RegularPolygon(
            (x + width/2, y + height/2), numVertices=4, radius=height*0.9, 
            orientation=0, linewidth=1.5, edgecolor=edgecolor, facecolor=color, zorder=2
        )
        ax.add_patch(rect)
    elif shape == 'cylinder':
        rect = patches.Ellipse((x + width/2, y + height), width, height*0.4, facecolor='lightgrey', edgecolor='black', zorder=3)
        ax.add_patch(rect)
        rect2 = patches.Rectangle((x, y + height*0.2), width, height*0.8, facecolor='lightgrey', edgecolor='black', zorder=2)
        ax.add_patch(rect2)
        rect3 = patches.Ellipse((x + width/2, y + height*0.2), width, height*0.4, facecolor='lightgrey', edgecolor='black', zorder=3)
        ax.add_patch(rect3)
        
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center', fontsize=9, fontweight='bold', zorder=4, wrap=True
    )

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color='black', mutation_scale=15),
        zorder=1
    )

def generate_diagram():
    # Increased figsize height slightly for better spacing
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110) # Increased Y-limit to provide ceiling for title
    ax.axis('off')

    # --- 1. Top Section (Shifted slightly down from 100 to avoid title) ---
    draw_box(ax, 40, 92, 20, 5, "Suspicious\nDocument", color='#FFE6CC') 
    draw_box(ax, 10, 82, 15, 8, "Reference\nCorpus", shape='cylinder')   
    draw_box(ax, 40, 80, 20, 5, "Preprocessing\n(NLTK)", color='#E6E6E6') 

    draw_arrow(ax, 50, 92, 50, 85)
    
    # --- 2. Split Streams ---
    draw_box(ax, 15, 62, 25, 6, "N-Gram\nExtraction", color='#DAE8FC')
    draw_box(ax, 15, 50, 25, 6, "Jaccard\nSimilarity", color='#DAE8FC')
    
    draw_box(ax, 60, 62, 25, 6, "SBERT\nBi-Encoder", color='#F8CECC')
    draw_box(ax, 60, 50, 25, 6, "Cosine\nSimilarity", color='#F8CECC')

    draw_arrow(ax, 50, 80, 27, 68) 
    draw_arrow(ax, 50, 80, 72, 68) 
    draw_arrow(ax, 20, 82, 27, 68) 
    draw_arrow(ax, 20, 82, 72, 68) 
    
    draw_arrow(ax, 27, 62, 27, 56) 
    draw_arrow(ax, 72, 62, 72, 56) 

    # --- 3. Decision Logic ---
    draw_box(ax, 41, 36, 18, 10, "Dual Threshold\n(Alpha, Beta)", shape='diamond', color='#FFF2CC')

    draw_arrow(ax, 27, 50, 41, 41) 
    draw_arrow(ax, 72, 50, 59, 41) 

    # --- 4. Forensics ---
    draw_arrow(ax, 50, 36, 50, 28)
    draw_box(ax, 35, 20, 30, 8, "Forensics Module\n(Passive Voice,\nSynonyms)", color='#D5E8D4')

    # --- 5. XAI & Output (Coordinates raised to stay above 0) ---
    draw_box(ax, 35, 10, 30, 6, "XAI Engine\n(Heatmap & Counterfactuals)", color='#E1D5E7')
    draw_box(ax, 32.5, 2, 35, 5, "Final Report", color='#FFE6CC')

    draw_arrow(ax, 50, 20, 50, 16)
    draw_arrow(ax, 50, 10, 50, 7)

    # Title with padding from the top box
    ax.text(50, 102, "HEPDS System Architecture", fontsize=16, fontweight='bold', ha='center')
    # Optional Figure Caption with gap

    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    print("✅ Optimized Diagram Generated!")

if __name__ == "__main__":
    generate_diagram()