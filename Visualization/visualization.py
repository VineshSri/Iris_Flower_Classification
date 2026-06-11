import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_feature_plot(df, output_path):
    """Save scatter plot of petal length vs petal width, colored by species."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 4))

    species = df["Species"].unique()
    colors = ["red", "green", "blue"]

    for sp, color in zip(species, colors):
        subset = df[df["Species"] == sp]
        plt.scatter(
            subset["PetalLengthCm"],
            subset["PetalWidthCm"],
            label=sp,
            color=color
        )

    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Iris Flower Features by Species")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Feature plot saved to {output_path}")


def save_confusion_matrix(matrix, labels, output_path):
    """Save confusion matrix as a heatmap image."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")
    
