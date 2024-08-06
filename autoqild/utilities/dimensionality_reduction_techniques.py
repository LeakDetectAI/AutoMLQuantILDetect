from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.manifold import TSNE


# Create a dictionary to store the techniques and their options
def create_dimensionality_reduction_model(reduction_technique, n_reduced=20):
    reduction_techniques = {
        'recursive_feature_elimination_et': RFE(ExtraTreesClassifier(), n_features_to_select=n_reduced),
        'recursive_feature_elimination_rf': RFE(RandomForestClassifier(), n_features_to_select=n_reduced),
        'select_from_model_et': SelectFromModel(ExtraTreesClassifier(), max_features=n_reduced),
        'select_from_model_rf': SelectFromModel(RandomForestClassifier(), max_features=n_reduced),
        'pca': PCA(n_components=n_reduced),
        'lda': LinearDiscriminantAnalysis(n_components=n_reduced),
        'tsne': TSNE(n_components=n_reduced),
        'nmf': NMF(n_components=n_reduced)
    }
    if reduction_technique not in reduction_techniques.keys():
        raise ValueError(f"Reduction type {reduction_technique} not defined {reduction_techniques.keys()}")
    selection_model = reduction_techniques[reduction_technique]
    return selection_model
