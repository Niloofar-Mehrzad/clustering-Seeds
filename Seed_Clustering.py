# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import pairwise
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix

# Load the dataset
file_path = 'C:/Users/mmehr/OneDrive/Desktop/CHALMERS/3_ThirdYear/3_StudyPeriod2/Introduction to data science and AI - DAT565/Assignments/Assignment_5/seeds.tsv'
data = pd.read_csv(file_path, sep='\t')

# Assign column names based on the dataset description
columns = [
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel Length",
    "Kernel Width",
    "Asymmetry Coefficient",
    "Kernel Groove Length",
    "Class Label"
]
data.columns = columns

# Separating features and the class label
features = data.drop(columns=["Class Label"])
class_label = data["Class Label"]

# Applying Min-Max scaling
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Adding the class label back to the normalized data
normalized_data = normalized_features.copy()
normalized_data["Class Label"] = class_label

# Display the first few rows of the normalized dataset
#print(normalized_data.head())



####### Problem 2 ########

# Drop the class label to only use features
features_only = normalized_data.drop(columns=["Class Label"])

# Define range of k values
k_values = range(1, 11)
inertias = []

# Perform k-means clustering for each k and compute inertia
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_only)
    inertias.append(kmeans.inertia_)

# Plot inertia vs k to determine the elbow point
"""plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Inertia vs. Number of Clusters (Elbow Method)')
plt.xticks(k_values)
plt.grid(True)
plt.show()"""


##### Problem 3 #####
#### Part a

# Pairwise scatter plots
plt.figure(figsize=(15, 15))
for i, column1 in enumerate(columns):
    for j, column2 in enumerate(columns):
        if i < j:  # Avoid duplicate plots (feature1 vs feature2 == feature2 vs feature1)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=normalized_data[column1],
                y=normalized_data[column2],
                hue=normalized_data["Class Label"],
                palette="viridis",
                legend="full"
            )
            plt.title(f"Scatter Plot: {column1} vs {column2}")
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.legend(title="Class Label", loc="best")
            plt.show()


"""
#####Part b
# Apply Gaussian Random Projection
def plot_grp_projection(data, n_components=2, random_seed=None):
    grp = GaussianRandomProjection(n_components=n_components, random_state=random_seed)
    projected_data = grp.fit_transform(data)

    # Plot the projected data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=normalized_data["Class Label"],
        cmap="viridis",
        edgecolor="k",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Class Label")
    plt.title("Gaussian Random Projection (2D)")
    plt.xlabel("Projected Dimension 1")
    plt.ylabel("Projected Dimension 2")
    plt.grid(True)
    plt.show()

# Call the function with your dataset
plot_grp_projection(normalized_data.drop(columns=["Class Label"]), random_seed=42)
"""
"""
#######Part c

# Apply UMAP for dimensionality reduction
def plot_umap_projection(data, labels, n_components=2, random_seed=None):
    umap_model = umap.UMAP(n_components=n_components, random_state=random_seed)
    umap_projection = umap_model.fit_transform(data)

    # Plot the UMAP projection
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_projection[:, 0],
        umap_projection[:, 1],
        c=labels,
        cmap="viridis",
        edgecolor="k",
        alpha=0.7
    )


    plt.colorbar(scatter, label="Class Label")
    plt.title("UMAP Projection (2D)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.show()  

# Prepare data and apply UMAP
features_only = normalized_data.drop(columns=["Class Label"])
class_labels = normalized_data["Class Label"]

plot_umap_projection(features_only, class_labels, random_seed=42)

####### Problem 4

# Function to compute Clustering Accuracy
def compute_accuracy(true_labels, predicted_labels):
    # Create a confusion matrix
    unique_classes = np.unique(true_labels)
    unique_clusters = np.unique(predicted_labels)
    n_classes = len(unique_classes)
    n_clusters = len(unique_clusters)
    confusion_matrix = np.zeros((n_classes, n_clusters), dtype=int)

    for i, class_label in enumerate(unique_classes):
        for j, cluster_label in enumerate(unique_clusters):
            confusion_matrix[i, j] = np.sum(
                (true_labels == class_label) & (predicted_labels == cluster_label)
            )

    # Solve the linear sum assignment problem to find the best label permutation
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Compute the accuracy based on the optimal assignment
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)

    return accuracy

# Perform k-means clustering for the chosen k (e.g., 3 from the elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(features_only)

# Compute Rand Index
rand_index = rand_score(class_label, kmeans_labels)
print(f"Rand Index: {rand_index:.4f}")

# Compute Clustering Accuracy
accuracy = compute_accuracy(class_label.to_numpy(), kmeans_labels)
print(f"Clustering Accuracy: {accuracy:.4f}")


########Problem 5##########

# Define a function to compute clustering accuracy
def compute_agglomerative_accuracy(true_labels, predicted_labels):
    unique_classes = np.unique(true_labels)
    unique_clusters = np.unique(predicted_labels)
    confusion_matrix = np.zeros((len(unique_classes), len(unique_clusters)))

    for i, cls in enumerate(unique_classes):
        for j, cluster in enumerate(unique_clusters):
            confusion_matrix[i, j] = np.sum((true_labels == cls) & (predicted_labels == cluster))

    # Solve the optimal label assignment problem
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy

# Define linkage options to test
linkage_options = ["ward", "complete", "average", "single"]
accuracies = {}

# Perform Agglomerative Clustering for each linkage method
for linkage_option in linkage_options:
    agglomerative = AgglomerativeClustering(n_clusters=3, linkage=linkage_option)
    predicted_labels = agglomerative.fit_predict(features_only)

    # Compute accuracy for the current linkage option
    accuracy = compute_agglomerative_accuracy(class_label.to_numpy(), predicted_labels)
    accuracies[linkage_option] = accuracy
    print(f"Linkage: {linkage_option}, Accuracy: {accuracy:.4f}")

# Find the best and worst linkage
best_linkage = max(accuracies, key=accuracies.get)
worst_linkage = min(accuracies, key=accuracies.get)
print(f"Best Linkage: {best_linkage}, Worst Linkage: {worst_linkage}")

# Plot dendrogram for the best linkage option
# Perform hierarchical clustering
linked = linkage(features_only, method=best_linkage)

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    labels=class_label.to_numpy(),
    leaf_rotation=90,
    leaf_font_size=10,
    color_threshold=0.7 * max(linked[:, 2]),  # Adjust threshold
)
plt.title(f"Dendrogram for Agglomerative Clustering (Linkage: {best_linkage})")
plt.xlabel("Sample Index or Class Label")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
"""