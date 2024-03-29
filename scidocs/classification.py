import ujson as json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from lightning.classification import LinearSVC
from sklearn.neural_network import MLPClassifier
from scidocs.embeddings import load_embeddings_from_jsonl


np.random.seed(1)


def get_mag_mesh_metrics(data_paths, embeddings_path=None, val_or_test='test', multifacet_behavior='concat', cls_svm=False, n_jobs=1):
    """Run MAG and MeSH tasks.

    Arguments:
        data_paths {scidocs.paths.DataPaths} -- A DataPaths objects that points to
                                                all of the SciDocs files

    Keyword Arguments:
        embeddings_path {str} -- Path to the embeddings jsonl (default: {None})
        val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                             or the test set (what's reported in SPECTER paper)

    Returns:
        metrics {dict} -- F1 score for both tasks.
    """
    assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"

    print('Loading MAG/MeSH embeddings...')

    embeddings = load_embeddings_from_jsonl(embeddings_path)

    print('Running the MAG task...')

    X, y, dim, num_facets = get_X_y_for_classification(embeddings, data_paths.mag_train, data_paths.mag_val, data_paths.mag_test, cls_svm=cls_svm)

    mag_f1 = classify(X['train'], y['train'], X[val_or_test], y[val_or_test], dim=dim, num_facets=num_facets, multifacet_behavior=multifacet_behavior, cls_svm=cls_svm, n_jobs=n_jobs)

    print('Running the MeSH task...')

    X, y, dim, num_facets = get_X_y_for_classification(embeddings, data_paths.mesh_train, data_paths.mesh_val, data_paths.mesh_test, cls_svm=cls_svm)

    mesh_f1 = classify(X['train'], y['train'], X[val_or_test], y[val_or_test], dim=dim, num_facets=num_facets, multifacet_behavior=multifacet_behavior, cls_svm=cls_svm, n_jobs=n_jobs)

    return {'mag': {'f1': mag_f1}, 'mesh': {'f1': mesh_f1}}


def classify(X_train, y_train, X_test, y_test, dim, num_facets, multifacet_behavior, cls_svm, n_jobs=1):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation.

    Arguments:
        X_train, y_train -- training data
        X_test, y_test -- test data to evaluate on (can also be validation data)

    Returns:
        F1 on X_test, y_test (out of 100), rounded to two decimal places
    """

    if cls_svm:
        estimator = LinearSVC(loss="squared_hinge", random_state=42)

        Cs = np.logspace(-4, 2, 7)

        svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)

        svm.fit(X_train, y_train)

        preds = svm.predict(X_test)
    else:
        # Instantiate a simple feedforward network using scikit-learn's MLPRegressor
        if multifacet_behavior == 'extra_linear':
            hidden_layer_sizes = (dim,)
        else:
            hidden_layer_sizes = ()

        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs', random_state=42, max_iter=3000, verbose=True)

        nn.fit(X_train, y_train)

        preds = nn.predict(X_test)

    return np.round(100 * f1_score(y_test, preds, average='macro'), 2)


def get_X_y_for_classification(embeddings, train_path, val_path, test_path, cls_svm=False):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair

    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings

    Returns:
        X, y: dictionaries of training X and training y
              with keys: 'train', 'val', 'test'
    """
    num_facets = len(next(iter(embeddings.values())))
    dim = len(next(iter(embeddings.values()))[0])

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    X = defaultdict(list)
    y = defaultdict(list)

    for dataset_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        for s2id, class_label in dataset.values:
            if s2id not in embeddings:
                if cls_svm:
                    X[dataset_name].append(np.zeros(dim))
                else:
                    X[dataset_name].append(np.zeros(dim * num_facets))
            else:
                if cls_svm and len(embeddings[s2id]) > 1:
                    X[dataset_name].append(embeddings[s2id].sum(axis=0))
                else:
                    X[dataset_name].append(embeddings[s2id].flatten())

            y[dataset_name].append(class_label)

        X[dataset_name] = np.array(X[dataset_name])
        y[dataset_name] = np.array(y[dataset_name])

    return X, y, dim, num_facets
