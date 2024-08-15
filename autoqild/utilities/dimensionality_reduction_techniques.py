from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.manifold import TSNE


# Create a dictionary to store the techniques and their options
def create_dimensionality_reduction_model(reduction_technique, n_reduced=20):
    """
    Creates a dimensionality reduction model based on the specified technique.

    Parameters
    ----------
    reduction_technique : str, optional, default=`select_from_model_rf`
        Technique to use for feature reduction, provided by scikit-learn.
        Must be one of:

        - `recursive_feature_elimination_et`: Uses ExtraTreesClassifier to recursively remove features and build a model.
        - `recursive_feature_elimination_rf`: Uses RandomForestClassifier to recursively remove features and build a model.
        - `select_from_model_et`: Meta-transformer for selecting features based on importance weights using ExtraTreesClassifier.
        - `select_from_model_rf`: Meta-transformer for selecting features based on importance weights using RandomForestClassifier.
        - `pca`: Principal Component Analysis for dimensionality reduction.
        - `lda`: Linear Discriminant Analysis for separating classes.
        - `tsne`: t-Distributed Stochastic Neighbor Embedding for visualization purposes.
        - `nmf`: Non-Negative Matrix Factorization for dimensionality reduction.
    n_reduced : int, optional
        The number of components or features to reduce to (default is 20).

    Returns
    -------
    selection_model: Dimensionality reduction Model
        A dimensionality reduction model corresponding to the specified technique.

    Raises
    ------
    ValueError
        If the specified reduction technique is not defined in {`recursive_feature_elimination_et`, `nmf`
        `recursive_feature_elimination_rf`, `select_from_model_et`, `select_from_model_rf`, `pca`, `lda`, `tsne`}
    """
    reduction_techniques = {
        `recursive_feature_elimination_et`: RFE(ExtraTreesClassifier(), n_features_to_select=n_reduced),
        `recursive_feature_elimination_rf`: RFE(RandomForestClassifier(), n_features_to_select=n_reduced),
        `select_from_model_et`: SelectFromModel(ExtraTreesClassifier(), max_features=n_reduced),
        `select_from_model_rf`: SelectFromModel(RandomForestClassifier(), max_features=n_reduced),
        `pca`: PCA(n_components=n_reduced),
        `lda`: LinearDiscriminantAnalysis(n_components=n_reduced),
        `tsne`: TSNE(n_components=n_reduced),
        `nmf`: NMF(n_components=n_reduced)
    }
    if reduction_technique not in reduction_techniques.keys():
        raise ValueError(f"Reduction type {reduction_technique} not defined {reduction_techniques.keys()}")
    selection_model = reduction_techniques[reduction_technique]
    return selection_model
