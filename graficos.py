import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    metrics = ["CIDEr", "BERTScore", "METEOR", "SPICE"]
    
    method_scores_satellite = {
        "Fine-Tune": [0.65, 0.50, 0.47, 0.2434],
        "B. Adapter": [0.68, 0.51, 0.49, 0.2530],
        "LoRA (A)": [0.63, 0.50, 0.46, 0.2294],
        "LoRA (L)": [0.68, 0.51, 0.48, 0.2417],
        "LayerNorm": [0.69, 0.51, 0.48, 0.2440],
        "WiSE-FT": [0.62, 0.48, 0.43, 0.2215],
    }
    
    method_scores_cub = {
        "Fine-Tune": [0.65, 0.50, 0.47, 0.2434],
        "B. Adapter": [0.68, 0.65, 0.51, 0.49, 0.2530],
        "LoRA (A)": [0.63, 0.50, 0.46, 0.2294],
        "LoRA (L)": [0.68, 0.51, 0.48, 0.2417],
        "LayerNorm": [0.69, 0.51, 0.48, 0.2440],
        "WiSE-FT": [0.62, 0.48, 0.43, 0.2215],
    }
    
    x = np.arange(len(metrics))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(layout="constrained", figsize=(12, 8))
    
    for attribute, measurement in method_scores_satellite.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
        
    ax.set_xticks(x + width, metrics)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 1.0)
    
    plt.savefig("metrics.png")


if __name__ == "__main__":
    main()
