from scidocs.paths import DataPaths
from scidocs import get_scidocs_metrics

import argparse
import ujson as json
import pandas as pd


pd.set_option('display.max_columns', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '--classification-embeddings-path', dest='cls', help='path to classification related embeddings (mesh and mag)')
    parser.add_argument('--user-citation', '--user_activity_and_citations_embeddings_path', dest='user_citation', help='path to user activity embeddings (coview, copdf, cocite, citation)')
    parser.add_argument('--recomm', '--recomm_embeddings_path', dest='recomm', help='path to recommender related embeddings')
    parser.add_argument('--val_or_test', default='test', help='whether to evaluate scidocs on test data (what is reported in the specter paper) or validation data (to tune hyperparameters)')

    parser.add_argument('--multifacet-behavior', default="concat", choices=['concat', 'extra_linear'], type=str)
    parser.add_argument('--cls-svm', default=False, action='store_true')
    parser.add_argument('--user-citation-normalize', default=False, action='store_true')

    parser.add_argument('--n-jobs', default=12, help='number of parallel jobs for classification (related to mesh/mag metrics)', type=int)
    parser.add_argument('--cuda-device', default=-1, help='specify if you want to use gpu for training the recommendation model; -1 means use cpu')
    parser.add_argument('--data-path', default=None, help='path to the data directory where scidocs files reside. If None, it will default to the `data/` directory')

    parser.add_argument('--results-save-path', default=None, help='path to the directory to save the results')

    args = parser.parse_args()

    data_paths = DataPaths(args.data_path)

    scidocs_metrics = get_scidocs_metrics(
        data_paths,
        args.cls,
        args.user_citation,
        args.recomm,
        val_or_test=args.val_or_test,
        multifacet_behavior=args.multifacet_behavior,
        cls_svm=args.cls_svm,
        user_citation_normalize=args.user_citation_normalize,
        n_jobs=args.n_jobs,
        cuda_device=args.cuda_device
    )

    flat_metrics = {}
    for k, v in scidocs_metrics.items():
        if not isinstance(v, dict):
            flat_metrics[k] = v
        else:
            for kk, vv in v.items():
                key = k + '-' + kk
                flat_metrics[key] = vv
    df = pd.DataFrame(list(flat_metrics.items())).T

    print(df)

    df.to_excel(args.results_save_path)

if __name__ == '__main__':
    main()




