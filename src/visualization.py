from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import warnings
from typing import Dict, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# Matplotlib Config
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA'
})


def largeDatasetOptimization(n_points: int, outlier_mask: np.ndarray) -> Tuple[float, int, int, bool]:
    """
    Optimize visualization parameters based on dataset size for better performance.
    
    Large datasets require different visualization strategies to maintain performance
    and readability. This function automatically adjusts point transparency, size,
    and sampling based on the number of data points.
    
    Performance tiers:
    - ≤1K points: Full visualization, high quality
    - ≤10K points: Moderate optimization
    - ≤50K points: Increased optimization
    - ≤500K points: Aggressive optimization with sampling
    - >500K points: Heavy sampling with outlier preservation
    
    Args:
        n_points (int): Total number of data points
        outlier_mask (np.ndarray): Boolean array indicating outliers
        
    Returns:
        Tuple[float, int, int, bool]: 
            - alpha: Point transparency (0-1)
            - point_size: Point size in plot
            - sample_size: Target number of points to display
            - should_sample: Whether to apply sampling
    """

    if n_points <= 1000:
        return 0.8, 60, n_points, False
    elif n_points <= 10000:
        return 0.6, 40, n_points, False
    elif n_points <= 50000:
        return 0.4, 25, n_points, False
    elif n_points <= 500000:
        return 0.3, 15, min(50000, n_points), True
    else:
        n_outliers = np.sum(outlier_mask)
        n_inliers_to_sample = min(30000, n_points - n_outliers)
        sample_size = n_outliers + n_inliers_to_sample
        return 0.25, 10, sample_size, True


def sampleData(data: np.ndarray, outlier_mask: np.ndarray, target_sample_size: int) -> np.ndarray:
    """
    Intelligently sample data while preserving all outliers.
    
    This function ensures that all outliers are retained in the visualization
    while randomly sampling from the inlier population to meet the target sample size.
    This approach maintains the outlier detection context while improving performance.
    
    Sampling strategy:
    1. Keep ALL outliers (essential for outlier detection visualization)
    2. Randomly sample inliers to fill remaining sample size
    3. If not enough inliers available, return all data points
    
    Args:
        data (np.ndarray): Original data array
        outlier_mask (np.ndarray): Boolean mask indicating outliers (True = outlier)
        target_sample_size (int): Desired number of points in sample
        
    Returns:
        np.ndarray: Indices of selected points for visualization
    """

    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(~outlier_mask)[0]
    n_outliers = len(outlier_indices)
    n_inliers_to_sample = min(target_sample_size - n_outliers, len(inlier_indices))
    
    if n_inliers_to_sample < len(inlier_indices):
        sampled_inlier_indices = np.random.choice(
            inlier_indices, 
            size=n_inliers_to_sample, 
            replace=False
        )
        selected_indices = np.concatenate([outlier_indices, sampled_inlier_indices])
    else:
        selected_indices = np.arange(len(data))
    
    return selected_indices


def addConvexHull(ax: plt.Axes, points: np.ndarray, color: str, 
                  alpha: float = 0.1, linewidth: float = 1) -> None:
    """
    Add a convex hull overlay to highlight data clusters.
    
    Convex hulls help visualize the boundary of data clusters and can reveal
    the spatial distribution of normal points vs outliers. This function
    safely handles edge cases like insufficient points or collinear data.
    
    Args:
        ax (plt.Axes): Matplotlib axes object to draw on
        points (np.ndarray): 2D array of points (n_points, 2)
        color (str): Color for the hull boundary and fill
        alpha (float, optional): Transparency for fill. Defaults to 0.1.
        linewidth (float, optional): Line width for boundary. Defaults to 1.
        
    Note:
        - Requires at least 3 non-collinear points to create a hull
        - Silently handles geometric edge cases (e.g., all points collinear)
        - Hull boundary is drawn with higher alpha than fill for visibility
    """

    if len(points) < 3:
        return
    
    try:
        hull = ConvexHull(points)
        
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 
                   color=color, alpha=alpha*3, linewidth=linewidth)
        
        hull_points = points[hull.vertices]
        hull_patch = mpatches.Polygon(hull_points, closed=True, facecolor=color, alpha=alpha, edgecolor=color, linewidth=linewidth)
        ax.add_patch(hull_patch)
    except:
        pass


def PCAvisualization(df: pd.DataFrame, outlierMask: np.ndarray, title: str, 
                    dim: int = 2, savePath: Optional[str] = None, dpi: int = 300, 
                    show_hulls: bool = True, show_density: bool = True) -> None:
    """
    Create publication-ready PCA visualization of outlier detection results.
    
    This function performs Principal Component Analysis to reduce high-dimensional
    data to 2D or 3D for visualization. It automatically optimizes performance for
    large datasets while maintaining visual quality and interpretability.
    
    Features:
    - Automatic performance optimization based on dataset size
    - Intelligent sampling that preserves all outliers
    - Optional convex hull overlays for cluster visualization  
    - Optional density plots for pattern recognition
    - Comprehensive statistics and variance explanation
    - Publication-ready styling and export options
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        outlierMask (np.ndarray): Boolean array indicating outliers (True = outlier)
        title (str): Plot title for the visualization
        dim (int, optional): Number of dimensions (2 or 3). Defaults to 2.
        savePath (Optional[str], optional): File path to save plot. Defaults to None.
        dpi (int, optional): Resolution for saved plot. Defaults to 300.
        show_hulls (bool, optional): Whether to show convex hulls. Defaults to True.
        show_density (bool, optional): Whether to show density plot. Defaults to True.
        
    Returns:
        None: Displays plot and optionally saves to file
        
    Raises:
        ValueError: If dim is not 2 or 3
    """

    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
        
    n_points = len(df)
    alpha, point_size, sample_size, should_sample = largeDatasetOptimization(n_points, outlierMask)
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(df)

    if should_sample and sample_size < n_points:
        selected_indices = sampleData(reduced, outlierMask, sample_size)
        reduced_sampled = reduced[selected_indices]
        mask_sampled = outlierMask[selected_indices]
        sample_info = f" (Showing {len(selected_indices):,} of {n_points:,} points)"
    else:
        reduced_sampled = reduced
        mask_sampled = outlierMask
        selected_indices = np.arange(n_points)
        sample_info = ""
    
    outliers = reduced_sampled[mask_sampled]
    inliers = reduced_sampled[~mask_sampled]
    explained_var = pca.explained_variance_ratio_
    
    if dim == 2:
        fig, ax = plt.subplots(figsize=(12, 9))
        if show_density and len(inliers) > 50:
            try:
                ax.hexbin(inliers[:, 0], inliers[:, 1], gridsize=30, 
                         cmap='Blues', alpha=0.3, mincnt=1)
            except:
                pass
        
        if len(inliers) > 0:
            scatter_inliers = ax.scatter(inliers[:, 0], inliers[:, 1], c='#4A90E2', s=point_size, alpha=alpha,
                                       label=f'Normal ({len(inliers):,})', edgecolors='white', linewidths=0.3)

        if len(outliers) > 0:
            scatter_outliers = ax.scatter(outliers[:, 0], outliers[:, 1], c='#E74C3C', s=point_size*1.5, 
                                        alpha=min(alpha*1.5, 1.0), label=f'Outliers ({len(outliers):,})', 
                                        edgecolors='darkred', linewidths=0.5, marker='D')
        
        if show_hulls:
            if len(inliers) > 2:
                addConvexHull(ax, inliers, '#4A90E2', alpha=0.1)
            if len(outliers) > 2:
                addConvexHull(ax, outliers, '#E74C3C', alpha=0.15)

        ax.set_xlabel(f'First Principal Component ({explained_var[0]:.1%} variance)', fontweight='medium')
        ax.set_ylabel(f'Second Principal Component ({explained_var[1]:.1%} variance)', fontweight='medium')
        
        stats_text = f"Total Variance Explained: {sum(explained_var):.1%}"
        if len(outliers) > 0:
            outlier_pct = len(outliers) / len(reduced_sampled) * 100
            stats_text += f"\nOutlier Rate: {outlier_pct:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='white', alpha=0.8, edgecolor='gray'))
        
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if len(inliers) > 0:
            ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='#4A90E2', s=point_size, alpha=alpha,
                      label=f'Normal ({len(inliers):,})', edgecolors='white', linewidths=0.2)

        if len(outliers) > 0:
            ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],  c='#E74C3C', s=point_size*1.5, alpha=min(alpha*1.5, 1.0),
                      label=f'Outliers ({len(outliers):,})', edgecolors='darkred', linewidths=0.3, marker='D')

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontweight='medium')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontweight='medium')
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})', fontweight='medium')

        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    plt.suptitle(f'{title}{sample_info}', fontsize=16, fontweight='bold', y=0.95)
    if len(inliers) > 0 and len(outliers) > 0:
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if savePath:
        plt.savefig(savePath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    plt.close()


def createSummaryViz(algorithm_results: Dict[str, np.ndarray], df_shape: Tuple[int, int], savePath: Optional[str] = None) -> None:
    """
    Create a summary bar chart comparing outlier detection algorithms.
    
    This function generates a comprehensive comparison visualization showing
    the number of outliers detected by different algorithms. It's useful for
    understanding algorithm behavior and choosing the most appropriate method
    for your data.
    
    Features:
    - Color-coded bars for easy algorithm comparison
    - Outlier counts displayed on top of each bar
    - Dataset information in the title
    - Publication-ready styling with customizable export options
    - Automatic layout optimization for algorithm names
    
    Args:
        algorithm_results (Dict[str, np.ndarray]): Dictionary mapping algorithm names 
            to boolean arrays indicating outliers
        df_shape (Tuple[int, int]): Shape of original dataset (n_samples, n_features)
        savePath (Optional[str], optional): File path to save plot. Defaults to None.
        
    Returns:
        None: Displays plot and optionally saves to file
    """

    algorithms = list(algorithm_results.keys())
    outlier_counts = [mask.sum() for mask in algorithm_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(algorithms, outlier_counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)

    for bar, count in zip(bars, outlier_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(outlier_counts)*0.01,
               f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Detection Algorithms', fontweight='medium')
    ax.set_ylabel('Number of Outliers Detected', fontweight='medium')
    ax.set_title(f'Outlier Detection Summary\nDataset: {df_shape[0]:,} samples × {df_shape[1]} features', 
                fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()

    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    plt.close()
