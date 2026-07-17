# ViPragSent Pragmatic-polarity Confusion Matrix

Status: `complete` · rows are gold labels · cells show count (row-normalised %).

| Gold \ Predicted | positive | negative | neutral | ironic-positive | sarcastic-negative | mocking |
| --- | ---: | ---: | ---: | ---: | ---: |
| positive | 224 (66.87%) | 8 (2.39%) | 102 (30.45%) | 0 (0.00%) | 0 (0.00%) | 1 (0.30%) |
| negative | 8 (3.36%) | 69 (28.99%) | 112 (47.06%) | 0 (0.00%) | 10 (4.20%) | 39 (16.39%) |
| neutral | 132 (13.29%) | 59 (5.94%) | 792 (79.76%) | 0 (0.00%) | 1 (0.10%) | 9 (0.91%) |
| ironic-positive | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| sarcastic-negative | 1 (0.78%) | 15 (11.63%) | 20 (15.50%) | 0 (0.00%) | 71 (55.04%) | 22 (17.05%) |
| mocking | 13 (4.26%) | 35 (11.48%) | 87 (28.52%) | 0 (0.00%) | 2 (0.66%) | 168 (55.08%) |

The fixed test split contains no `ironic-positive` pragmatic-polarity records, so its row has no denominator. See `../figures/fig5_confusion.svg` for the corresponding heatmap.

Source: `results/error_confusion.json`.
