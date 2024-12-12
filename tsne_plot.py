import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_annotated_tsne(features, labels, max_samples=10000, pca_components=300,
                          concentrations_to_annotate=[0, 50, 100, 300, 500]):
    """
    Create t-SNE visualization with annotations for specific concentrations
    """
    # Convert to numpy arrays
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # If dataset is too large, take a stratified subset
    if len(features_np) > max_samples:
        print(f"Reducing dataset size from {len(features_np)} to {max_samples} samples...")

        unique_labels = np.unique(labels_np)
        indices = []
        samples_per_class = max_samples // len(unique_labels)

        for label in unique_labels:
            label_indices = np.where(labels_np == label)[0]
            if len(label_indices) > samples_per_class:
                selected_indices = np.random.choice(label_indices, samples_per_class, replace=False)
            else:
                selected_indices = label_indices
            indices.extend(selected_indices)

        np.random.shuffle(indices)
        features_np = features_np[indices]
        labels_np = labels_np[indices]

    # Apply PCA
    print(f"Reducing dimensions from {features_np.shape[1]} to {pca_components} using PCA...")
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features_np)

    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2,
                perplexity=min(30, len(features_pca) - 1),
                n_iter=10000,
                random_state=42,
                verbose=1)
    tsne_results = tsne.fit_transform(features_pca)

    # Create visualization
    plt.figure(figsize=(8, 6))

    # Create color map
    unique_labels = np.sort(np.unique(labels_np))
    norm = plt.Normalize(unique_labels.min(), unique_labels.max())

    # Create scatter plot
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                          c=labels_np,
                          cmap='viridis',
                          norm=norm,
                          s=40,
                          alpha=0.8)

    # # Add annotations for specific concentrations
    # for concentration in concentrations_to_annotate:
    #     # Find points with this concentration
    #     concentration_mask = (labels_np == concentration)
    #     if np.any(concentration_mask):
    #         # Get the center point for this concentration
    #         points = tsne_results[concentration_mask]
    #         center = points.mean(axis=0)
    #
    #         # Find the point closest to the center
    #         distances = np.sum((points - center) ** 2, axis=1)
    #         closest_point_idx = np.argmin(distances)
    #         annotation_point = points[closest_point_idx]
    #
    #         # Add annotation with arrow
    #         plt.annotate(f'{concentration} mg/L',
    #                      xy=(annotation_point[0], annotation_point[1]),
    #                      xytext=(annotation_point[0] + 2, annotation_point[1] + 2),
    #                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
    #                      arrowprops=dict(arrowstyle='->',
    #                                      connectionstyle='arc3,rad=0.2',
    #                                      color='black'))

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Concentration (mg/L)', rotation=270, labelpad=15)

    # Add labels and title
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.grid(False)

    # Save the figure with 1200 dpi
    # plt.savefig('tsne_scatter_plot.png', dpi=1200)

    # # Store points for each annotated concentration
    # annotated_points = {}
    # for concentration in concentrations_to_annotate:
    #     mask = (labels_np == concentration)
    #     if np.any(mask):
    #         annotated_points[concentration] = {
    #             'points': tsne_results[mask],
    #             'indices': np.where(mask)[0]
    #         }

    annotated_points = None
    return tsne_results, indices if len(features_np) > max_samples else None, annotated_points


def main(features_path):
    """
    Load features and create visualizations
    """
    # Load features and labels
    print("Loading features...")
    feature_dict = torch.load(features_path)
    features = feature_dict['features']
    labels = feature_dict['labels']

    # Stack and normalize features
    print("Processing features...")
    all_features = []
    for feature in tqdm(features):
        # Flatten the feature
        feature = feature.view(-1)
        all_features.append(feature)

    all_features = torch.stack(all_features)
    labels = torch.stack(labels)

    # Normalize features
    def min_max_normalize(features):
        min_val = torch.min(features)
        max_val = torch.max(features)
        return (features - min_val) / (max_val - min_val)

    print("Normalizing features...")
    all_features = min_max_normalize(all_features)

    # Create visualizations
    print("Creating visualizations...")

    tsne_results, indices, annotated_points = create_annotated_tsne(
        all_features,
        labels,
        max_samples=10000,
        pca_components=305,
        concentrations_to_annotate=[0, 50, 100, 300, 500]
    )
    plt.show()

    print("Visualization complete!")


if __name__ == "__main__":
    features_path = 'deep_features/mobilenetv3/NACO_ANCO_split_test.pt'
    main(features_path)