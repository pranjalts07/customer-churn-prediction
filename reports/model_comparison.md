# Model Comparison Report

The comparison uses the existing processed training and test sets in `data/processed/`.
Each candidate is trained on the training set and evaluated on the held out test set.
SMOTE is applied inside each training pipeline to handle class imbalance.

Business value uses this formula:

```text
true positives * 150 - flagged customers * 25
```

Best model by business value: Logistic Regression.

| model                |   auc_roc |   auc_pr |   best_threshold |   business_value |   precision_at_best_threshold |   recall_at_best_threshold |   f1_at_best_threshold |
|:---------------------|----------:|---------:|-----------------:|-----------------:|------------------------------:|---------------------------:|-----------------------:|
| Logistic Regression  |    0.8024 |   0.2545 |           0.2800 |        3575.0000 |                        0.2387 |                     0.5766 |                 0.3376 |
| Gradient Boosting    |    0.7933 |   0.2457 |           0.2700 |        3425.0000 |                        0.2332 |                     0.5839 |                 0.3333 |
| Random Forest        |    0.7787 |   0.2184 |           0.3400 |        3050.0000 |                        0.2349 |                     0.5109 |                 0.3218 |
| AdaBoost             |    0.7963 |   0.2360 |           0.4800 |        3000.0000 |                        0.2708 |                     0.3796 |                 0.3161 |
| Decision Tree        |    0.7266 |   0.2235 |           0.3000 |        2975.0000 |                        0.2533 |                     0.4234 |                 0.3169 |
| KNN                  |    0.7811 |   0.2143 |           0.3200 |        2450.0000 |                        0.2052 |                     0.6350 |                 0.3102 |
| Extra Trees          |    0.7794 |   0.2159 |           0.3800 |        2375.0000 |                        0.2487 |                     0.3504 |                 0.2909 |
| Gaussian Naive Bayes |    0.7881 |   0.2202 |           0.5300 |        2200.0000 |                        0.2761 |                     0.2701 |                 0.2731 |

Note: the threshold is selected to maximize business value on the held out comparison set.
For production retraining, choose the threshold on a validation set and keep the test set for final reporting.
