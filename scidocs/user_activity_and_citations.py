import os
import operator
import pathlib
import tempfile

import pytrec_eval
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scidocs.embeddings import load_embeddings_from_jsonl


def get_view_cite_read_metrics(data_paths, embeddings_path=None, val_or_test='test', multifacet_behavior='concat',
                               user_citation_normalize=False, user_citation_metric="l2", tasks_to_run=['cite', 'cocite', 'coview', 'coread'], temp_save_path=None):
    """Run the cocite, coread, coview, cite task evaluations.

    Arguments:
        data_paths {scidocs.paths.DataPaths} -- A DataPaths objects that points to
                                                all of the SciDocs files

    Keyword Arguments:
        embeddings_path {str} -- Path to the embeddings jsonl (default: {None})
        val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                             or the test set (what's reported in SPECTER paper)

    Returns:
        metrics {dict} -- NDCG and MAP for all four tasks.
    """
    assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"

    print('Loading co-view, co-read, cite, and co-cite embeddings...')
    embeddings = load_embeddings_from_jsonl(embeddings_path)

    if temp_save_path is not None:
        run_path = os.path.join(temp_save_path, 'temp.run')
    else:
        run_path = os.path.join(tempfile.mkdtemp(), 'temp.run')

    print('Running the co-view, co-read, cite, and co-cite tasks...')
    if val_or_test == 'test':
        if 'coview' in tasks_to_run:
            make_run_from_embeddings(data_paths.coview_test, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            coview_results = qrel_metrics(data_paths.coview_test, run_path, metrics=('ndcg', 'map'))

        if 'coread' in tasks_to_run:
            make_run_from_embeddings(data_paths.coread_test, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            coread_results = qrel_metrics(data_paths.coread_test, run_path, metrics=('ndcg', 'map'))

        if 'cite' in tasks_to_run:
            make_run_from_embeddings(data_paths.cite_test, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior,  user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            cite_results = qrel_metrics(data_paths.cite_test, run_path, metrics=('ndcg', 'map'))

        if 'cocite' in tasks_to_run:
            make_run_from_embeddings(data_paths.cocite_test, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            cocite_results = qrel_metrics(data_paths.cocite_test, run_path, metrics=('ndcg', 'map'))
    elif val_or_test == 'val':
        if 'coview' in tasks_to_run:
            make_run_from_embeddings(data_paths.coview_val, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            coview_results = qrel_metrics(data_paths.coview_val, run_path, metrics=('ndcg', 'map'))

        if 'coread' in tasks_to_run:
            make_run_from_embeddings(data_paths.coread_val, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            coread_results = qrel_metrics(data_paths.coread_val, run_path, metrics=('ndcg', 'map'))

        if 'cite' in tasks_to_run:
            make_run_from_embeddings(data_paths.cite_val, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            cite_results = qrel_metrics(data_paths.cite_val, run_path, metrics=('ndcg', 'map'))

        if 'cocite' in tasks_to_run:
            make_run_from_embeddings(data_paths.cocite_val, embeddings, run_path, topk=5, multifacet_behavior=multifacet_behavior, user_citation_normalize=user_citation_normalize, user_citation_metric=user_citation_metric, generate_random_embeddings=False)
            cocite_results = qrel_metrics(data_paths.cocite_val, run_path, metrics=('ndcg', 'map'))

    all_results = {}

    if 'coview' in tasks_to_run:
        all_results['co-view'] = coview_results

    if 'coread' in tasks_to_run:
        all_results['co-read'] = coread_results

    if 'cite' in tasks_to_run:
        all_results['cite'] = cite_results

    if 'cocite' in tasks_to_run:
        all_results['co-cite'] = cocite_results

    return all_results


def qrel_metrics(qrel_file, run_file, metrics=('ndcg', 'map')):
    """Get metrics (ndcg and map by default) for a run compared to a qrel file.

    Arguments:
        qrel_file -- qrel file with ground truth data
        run_file -- predictions from the run
        metrics -- which metrics to evaluate on,
                   can use any valid metrics that the trec_eval tool accepts

    Returns:
        metric_values -- dictionary of metric values (out of 100), rounded to two decimal places
    """
    with open(qrel_file, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(run_file, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics))
    results = evaluator.evaluate(run)

    metric_values = {}

    for measure in sorted(metrics):
        res = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]  for query_measures in results.values()]
            )
        metric_values[measure] = np.round(100 * res, 2)
    return metric_values


def make_run_from_embeddings(qrel_file, embeddings, run_file, topk=5, multifacet_behavior='concat', user_citation_normalize=False, user_citation_metric="l2", generate_random_embeddings=False):
    """Given embeddings and a qrel file, construct a run file.

    Arguments:
        qrel_file -- qrel file with ground truth data
        embeddings -- dictionary of embeddings
        run_file -- where to put the run file that is generated
        topk -- how many of the top nearest neighbors to write
        generate_random_embeddings -- whether to just use random emebddings and ignore the `embeddings` variable


    Returns:
        None
    """
    with open(qrel_file) as f_in:
        qrels = [line.strip() for line in f_in]

    # a dict where keys are paper-ids and values are all relevant and
    # non-relevant paper-ids in the qrel
    papers = defaultdict(list)

    # each row is in the following format
    # query-id 0 paper-id [relevance]
    # where relevance is 0=negative or 1=positive
    for line in qrels:
        row = line.split(' ')
        papers[row[0]].append(row[2])

    results = []

    missing_queries = 0
    key_error = 0
    success_candidates = 0
    for pid in papers:
        try:
            if generate_random_embeddings:
                emb_query = np.random.normal(0, 0.67, 200)
            else:
                if multifacet_behavior == 'extra_linear':
                    emb_query = embeddings[pid]
                else:
                    emb_query = embeddings[pid].flatten()

                if user_citation_normalize:
                    emb_query = normalize(emb_query, norm="l2", axis=1)

        except KeyError:
            missing_queries += 1
            continue

        if len(emb_query) == 0:
            missing_queries += 1
            continue

        # all embeddings for candidate paper ids in the qrel file
        emb_candidates = []
        candidate_ids = []
        for idx, paper_id in enumerate(papers[pid]):
            try:
                if generate_random_embeddings:
                    emb_candidates.append(np.random.normal(0, 0.67, 200))
                else:
                    if multifacet_behavior == 'extra_linear':
                        candidate_embeddings = embeddings[paper_id]
                    else:
                        candidate_embeddings = embeddings[paper_id].flatten()

                    if user_citation_normalize:
                        candidate_embeddings = normalize(candidate_embeddings, norm="l2", axis=1)

                    emb_candidates.append(candidate_embeddings)

                candidate_ids.append(paper_id)
                success_candidates += 1
            except KeyError:
                key_error += 1

        if multifacet_behavior == 'extra_linear':
            # Record distance calculated between all combinations of
            # facet embeddings from query AND
            # facet embeddings from each candidate paper.
            # Note: This list should have the same ordering as emb_candidates.
            distances = []

            for e in emb_candidates:
                if len(e) > 0:
                    all_possible_distances = []

                    if user_citation_metric == "dot":
                        all_possible_distances = np.dot(emb_query, e.T)
                    elif user_citation_metric == "cosine":
                        all_possible_distances = cosine_similarity(emb_query, e)
                    else:
                        all_possible_distances = euclidean_distances(emb_query, e)

                    # The highest score achieved by one of the embeddings
                    # should be used as the final score.
                    if user_citation_metric in ("dot", "cosine"):
                        # For dot product and cosine similarity, simply choose the biggest dot product/similarity value obtained.
                        distances.append(np.max(all_possible_distances))
                    else:
                        # For L2, the highest score is obtained by negating minimum distance.
                        distances.append(-np.min(all_possible_distances))
                else:
                    distances.append(float("-inf"))
        else:
            if user_citation_metric == "dot":
                distances = [np.dot(emb_query, e.T)[0][0]
                            if len(e) > 0 else float("-inf")
                            for e in emb_candidates]
            elif user_citation_metric == "cosine":
                distances = [cosine_similarity(emb_query, e.T)[0][0]
                            if len(e) > 0 else float("-inf")
                            for e in emb_candidates]
            else:
                # trec_eval assumes higher scores are more relevant
                # here the closer distance means higher relevance; therefore, we multiply distances by -1
                distances = [-euclidean_distances(emb_query, e)[0][0]
                            if len(e) > 0 else float("-inf")
                            for e in emb_candidates]

        distance_with_ids = list(zip(candidate_ids, distances))

        sorted_dists = sorted(distance_with_ids, key=operator.itemgetter(1))

        added = set()
        for i in range(len(sorted_dists)):
            # output is in this format: [qid iter paperid rank similarity run_id]
            if sorted_dists[i][0] in added:
                continue
            if i < len(sorted_dists) - topk:
                results.append([pid, '0', sorted_dists[i][0], '0', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            else:
                results.append([pid, '0', sorted_dists[i][0], '1', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            added.add(sorted_dists[i][0])

    pathlib.Path(run_file).parent.mkdir(parents=True, exist_ok=True)

    with open(run_file, 'w') as f_out:
        for res in results:
            f_out.write(f"{' '.join(res)}\n")
