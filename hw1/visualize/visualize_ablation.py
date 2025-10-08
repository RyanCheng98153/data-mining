import re
import matplotlib.pyplot as plt

def parse_ablation_log(log_path):
    """
    Parse ablation results from the log file.
    Extracts final RMSE values for baseline and each feature-removed model.
    """
    results = {}
    current_feature = "Baseline (all features)"
    current_rmse = None

    with open(log_path, 'r', encoding='utf-16') as f:
        for line in f:
            # Detect new section
            feature_match = re.search(r"Running without feature: (.+)", line)
            if feature_match:
                if current_rmse is not None:
                    results[current_feature] = current_rmse
                current_feature = feature_match.group(1).strip()
                current_rmse = None

            # Detect final RMSE
            rmse_match = re.search(r"Epoch\s+20000 train_rmse=([\d\.]+)", line)
            if rmse_match:
                current_rmse = float(rmse_match.group(1))

        # Save last one
        if current_rmse is not None:
            results[current_feature] = current_rmse

    return results


def visualize_ablation(results, save_path="ablation_results.png"):
    """
    Draws a bar chart comparing baseline vs ablations.
    Adds baseline as the first bar, RMSE labels below, and ΔRMSE above.
    """
    baseline_rmse = results.get("Baseline (all features)", None)

    # Collect ablations and sort by RMSE difference
    ablations = [(k, v, v - baseline_rmse) for k, v in results.items() if k != "Baseline (all features)"]
    ablations.sort(key=lambda x: x[2], reverse=True)

    features = ["Baseline \n(all features)"] + [a[0] for a in ablations]
    rmses = [baseline_rmse] + [a[1] for a in ablations]
    diffs = [0.0] + [a[2] for a in ablations]

    plt.figure(figsize=(12, 6))
    colors = ['skyblue'] + ['lightcoral'] * len(ablations)
    bars = plt.bar(features, rmses, color=colors)

    # Draw baseline reference line
    plt.axhline(y=baseline_rmse, color='blue', linestyle='--', linewidth=1, label=f'Baseline RMSE = {baseline_rmse:.4f}')

    # Add RMSE (below bar) and ΔRMSE (above bar)
    for i, (rmse, diff) in enumerate(zip(rmses, diffs)):
        # 下方顯示 RMSE
        plt.text(i, rmse - 0.05, f"{rmse:.4f}", ha='center', va='top', fontsize=9, color='black')
        # 上方顯示差值（Baseline 不顯示 +）
        if i == 0:
            plt.text(i, rmse + 0.05, "Baseline", ha='center', va='bottom', fontsize=9, color='black')
        else:
            plt.text(i, rmse + 0.05, f"+{diff:.4f}", ha='center', va='bottom', fontsize=9, color='black')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Final Train RMSE')
    plt.title('Ablation Study Results (Remove One Feature)')
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"✅ Figure saved as {save_path}")
    plt.show()


if __name__ == "__main__":
    log_path = "./ablation.log.result"  # or your actual log path
    results = parse_ablation_log(log_path)
    print("Parsed results:")
    for k, v in results.items():
        print(f"{k:30s} -> RMSE = {v:.4f}")
    visualize_ablation(results)
