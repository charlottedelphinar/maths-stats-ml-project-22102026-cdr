# Old decision tree (before the wider search and the PageValues option)

This folder keeps our decision tree project as it was before the later changes.

## Code
The files are the original ones. Only two things changed so they can run inside this folder:
the imports now say `old_decision_tree.` instead of `src.`, and the graphs are saved in `old_visuals/`.
The maths and the settings are untouched (3 depths, 5 purchase weights, minimum leaf 20).

Run from the project root:

    python -m old_decision_tree.final_test
    python -m old_decision_tree.visualize_results

Expected test result: equal weights F1 0.1670, weighted purchases (depth 5, weight 5.5702) precision 0.3132, recall 0.6438, F1 0.4214.

## old_visuals
The graphs from that version, with new names:
- `old_model_comparison_depth_vs_f1.png` (was `results/model_comparison.png`)
- `old_target_distribution.png`

## test_history
The notebooks we used to test the tree, in the order we made them. See `test_history/README.md`.
