from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def PCAvisualization(df, outlier_mask, title, dim=2, savePath=None):
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(df)
    plt.figure(figsize=(10, 8))

    if dim == 2:
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=outlier_mask, palette={True: 'red', False: 'blue'}, legend=False)
    else:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=['red' if o else 'blue' for o in outlier_mask])

    plt.title(title)
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath)

    plt.close()
