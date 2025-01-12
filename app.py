import streamlit as st
import numpy as np
from PIL import Image

# -------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------

def load_image(image_file) -> np.ndarray:
    """
    Convert uploaded file to a NumPy array (RGB).
    """
    image = Image.open(image_file).convert("RGB")  # ensure RGB
    return np.array(image)

def image_to_data(image: np.ndarray) -> np.ndarray:
    """
    Flatten the image array from (H, W, 3) to (H*W, 3).
    """
    return image.reshape((-1, 3))

def data_to_image(data: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """
    Reshape from (H*W, 3) back to (H, W, 3).
    """
    return data.reshape(shape)

def sample_data(data: np.ndarray, sample_size: int = 10000) -> np.ndarray:
    """
    Randomly sample up to 'sample_size' points from 'data'.
    Useful for speeding up K-Means on very large images.
    """
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        return data[indices]
    return data

# -------------------------------------------------------
# 2. K-MEANS INITIALIZATION
# -------------------------------------------------------
def initialize_centroids_kmeans(data: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    K-Means centroid initialization.
    Chooses the first centroid randomly, then chooses subsequent
    centroids with probability proportional to the squared distance from existing centroids.
    """
    np.random.seed(random_state)
    centroids = []

    # Choose the first centroid randomly
    first_idx = np.random.choice(len(data))
    centroids.append(data[first_idx])

    # Choose remaining centroids
    for _ in range(k - 1):
        # Compute distances to the closest existing centroid
        dist_sq = np.min(
            np.linalg.norm(data - np.array(centroids)[:, np.newaxis], axis=2) ** 2,
            axis=0
        )
        # Probability is proportional to distance squared
        probabilities = dist_sq / dist_sq.sum()
        chosen_idx = np.random.choice(len(data), p=probabilities)
        centroids.append(data[chosen_idx])

    return np.array(centroids)

# -------------------------------------------------------
# 3. CORE K-MEANS FUNCTIONS
# -------------------------------------------------------
def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest centroid.
    Returns an array of cluster indices for each data point.
    """
    # distances.shape => (k, len(data))
    distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
    # clusters.shape => (len(data),)
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(data: np.ndarray, clusters: np.ndarray, k: int) -> np.ndarray:
    """
    Recompute the centroids by taking the mean of all data points assigned to each cluster.
    If a cluster has no points, re-initialize it randomly from the data.
    """
    new_centroids = []
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) == 0:
            # Re-initialize this centroid randomly to avoid empty cluster
            random_idx = np.random.choice(len(data))
            new_centroids.append(data[random_idx])
        else:
            new_centroids.append(np.mean(points_in_cluster, axis=0))
    return np.array(new_centroids)

def kmeans_from_scratch(
    data: np.ndarray,
    k: int,
    max_iter: int = 100,
    epsilon: float = 1e-2,
    use_kmeans: bool = True,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform K-Means clustering on 'data' with 'k' clusters.

    Parameters
    ----------
    data : np.ndarray
        2D array of data points (num_points, num_features).
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations.
    epsilon : float
        Threshold for centroid movement to decide convergence.
    use_kmeans : bool
        If True, use K-Means initialization. Otherwise, random initialization.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    centroids : np.ndarray
        Final centroid positions.
    clusters : np.ndarray
        Cluster assignment for each data point.
    """
    # 1. Initialize centroids
    if use_kmeans:
        centroids = initialize_centroids_kmeans(data, k, random_state)
    else:
        np.random.seed(random_state)
        random_idx = np.random.choice(len(data), size=k, replace=False)
        centroids = data[random_idx]

    # 2. Iteration loop
    for _ in range(max_iter):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # 3. Check for convergence via threshold
        centroid_shift = np.linalg.norm(centroids - new_centroids, axis=1).max()
        if centroid_shift < epsilon:
            break
        centroids = new_centroids

    return centroids, clusters

# -------------------------------------------------------
# 4. STREAMLIT APP
# -------------------------------------------------------
def main():
    st.title("K-Means Color Paletter Extraction from Scratch! ")
    st.write(
        "Upload an image, choose the number of colors, sampling, and other parameters. "
        "We'll cluster the pixel colors with a custom K-Means implementation."
    )

    # Sidebar: K-Means parameters
    st.sidebar.header("Parameters")

    n_colors = st.sidebar.slider("Number of Colors (k)", 2, 20, 5)
    max_iter = st.sidebar.number_input("Max Iterations", min_value=1, max_value=1000, value=100)
    epsilon = st.sidebar.slider("Convergence Threshold (epsilon)", 1e-4, 1.0, 0.01, format="%.4f")

    # Optional sampling
    sampling_enabled = st.sidebar.checkbox("Enable Sampling (faster on large images)", value=True)
    sample_size = st.sidebar.number_input("Sample Size", 1000, 50000, 10000, step=1000)

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)  # (H, W, 3)
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        if st.button("Extract Palette"):
            data = image_to_data(image).astype(float)

            # Sample data if enabled
            if sampling_enabled:
                data_for_kmeans = sample_data(data, sample_size)
            else:
                data_for_kmeans = data

            # Run K-Means
            centroids, clusters = kmeans_from_scratch(
                data_for_kmeans,
                k=n_colors,
                max_iter=max_iter,
                epsilon=epsilon,
                use_kmeans=True,
                random_state=42
            )

            # If sampling was enabled, we now need to assign ALL pixels to these centroids
            # Re-assign clusters on the full dataset using the final centroids
            full_clusters = assign_clusters(data, centroids)

            # Round centroids for nicer display
            centroids_rounded = np.rint(centroids).astype(int).clip(0, 255)

            # Show the color palette
            st.subheader("Extracted Color Palette")
            palette = np.zeros((50, 50 * n_colors, 3), dtype=int)
            for i, c in enumerate(centroids_rounded):
                palette[:, i * 50:(i + 1) * 50, :] = c
            st.image(palette, caption="Color Palette", use_container_width=True)

            # Reconstruct the image using the cluster assignments
            reconstructed_data = np.array([centroids_rounded[c] for c in full_clusters])
            reconstructed_image = data_to_image(reconstructed_data, image.shape)

            st.subheader("Reconstructed Image")
            st.image(reconstructed_image.astype(np.uint8), use_container_width=True)

if __name__ == "__main__":
    main()
