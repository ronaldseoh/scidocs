# SciDocs

Please also take a look at [the original SciDocs README](README_original.md) for the introduction to SciDocs.

This repo contains our modifications to the original SciDocs code to allow 1) multiple embeddings and 2) our own `cite` and `co-cite` data.

The command for running the original set of tests in SciDocs:

```bash
python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --data-path scidocs/data \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --user-citation-metric "cosine" \
                      --results-save-path save/results_cosine.xlsx
```

The command for running evaluating with custom `cite` and `co-cite` data:

```bash
python ../scidocs/scripts/run_custom_cite.py --user-citation user-citation_custom_cite.jsonl \
                      --data-path scidocs-shard7 \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --results-save-path save/results_cosine_custom_cite.xlsx
```

Please adjust the paths accordingly.
