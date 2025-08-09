from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import warnings

warnings.filterwarnings("ignore")

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

def largeDatasetOptimization(n_points, outlier_mask):
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

def sampleData(data, outlier_mask, target_sample_size):
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

def addConvexHull(ax, points, color, alpha=0.1, linewidth=1):
    if len(points) < 3:
        return
    
    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=alpha*3, linewidth=linewidth)
        hull_points = points[hull.vertices]
        hull_patch = mpatches.Polygon(hull_points, closed=True, facecolor=color, alpha=alpha, edgecolor=color, linewidth=linewidth)
        ax.add_patch(hull_patch)
    except:
        pass

def PCAvisualization(df: pd.DataFrame, outlierMask: np.ndarray, title: str, dim: int = 2, savePath: str = None, dpi: int = 300, show_hulls: bool = True, show_density: bool = True) -> None:
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
            scatter_outliers = ax.scatter(outliers[:, 0], outliers[:, 1], c='#E74C3C', s=point_size*1.5, alpha=min(alpha*1.5, 1.0),
                                          label=f'Outliers ({len(outliers):,})', edgecolors='darkred', linewidths=0.5, marker='D')
        
        if show_hulls:
            if len(inliers) > 2:
                addConvexHull(ax, inliers, '#4A90E2', alpha=0.1)
            if len(outliers) > 2:
                addConvexHull(ax, outliers, '#E74C3C', alpha=0.15)

        ax.set_xlabel(f'First Principal Component ({explained_var[0]:.1%} variance)', 
                     fontweight='medium')
        ax.set_ylabel(f'Second Principal Component ({explained_var[1]:.1%} variance)', 
                     fontweight='medium')
        
        stats_text = f"Total Variance Explained: {sum(explained_var):.1%}"
        if len(outliers) > 0:
            outlier_pct = len(outliers) / len(reduced_sampled) * 100
            stats_text += f"\nOutlier Rate: {outlier_pct:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='white', alpha=0.8, edgecolor='gray'))
        
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if len(inliers) > 0:
            ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='#4A90E2', s=point_size, alpha=alpha,
                      label=f'Normal ({len(inliers):,})', edgecolors='white', linewidths=0.2)

        if len(outliers) > 0:
            ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='#E74C3C', s=point_size*1.5, alpha=min(alpha*1.5, 1.0),
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
        plt.savefig(savePath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()

def createSummaryViz(algorithm_results: dict, df_shape: tuple, 
                               savePath: str = None) -> None:
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
        plt.savefig(savePath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
