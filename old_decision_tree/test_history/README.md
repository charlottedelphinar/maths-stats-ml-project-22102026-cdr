# History of our decision tree tests

The notebooks are in the order we made them. All use the same data, the same seed 42 split (60/20/20)
and the same 16 features (without PageValues).

| File | What we tested | What we found |
|---|---|---|
| `01_full_research_equal_weights.ipynb` | Our tree against scikit-learn, depths 3, 5, 7, equal weights | Best validation F1 0.1732 (depth 7). Test: precision 0.4000, recall 0.1055, F1 0.1670 |
| `02_weighted_decision_tree_colab.ipynb` | Giving purchases a higher weight, 30 configurations | Weighting raised recall a lot. Test F1 went from 0.1670 to 0.4214 (depth 5, weight 5.5702) |
| `03_final_weighted_tree_selection.ipynb` | Same 30 configurations, selecting by precision and recall of at least 0.35 | No configuration reached both on validation, so it used the best F1 (0.4354 on validation) |
| `04_precision_recall_threshold_search.ipynb` | Adding a decision threshold on the leaf score, targets of 0.50 for both | Targets not reached on validation. Test: precision 0.3308, recall 0.5726, F1 0.4193. Full output in `04_precision_recall_threshold_search_results.txt` |

Notebooks 02 to 04 are written for Google Colab (they ask for the CSV upload).

What came after this history is in the main project: the wider search in `src/`, and the
logistic regression and neural network approaches in `logistic_regression/` and `neural_network/`.
