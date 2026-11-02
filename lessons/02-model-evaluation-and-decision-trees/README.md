# Lesson 02 — Model Evaluation and Decision Trees

These notes capture my understanding of the second class session: K-fold cross-validation, classification metrics, the difference between instance-based and model-based learning, and how decision trees use impurity to choose splits.

## 1. K-fold cross-validation

K-fold cross-validation estimates how well a model is likely to generalize without depending on one lucky or unlucky validation split.

1. Divide the **development data** into `K` approximately equal folds.
2. Train on `K - 1` folds and validate on the remaining fold.
3. Repeat until every fold has been the validation fold once.
4. Report the mean score, and preferably its standard deviation.

For five-fold CV, each model trains on about 80% of the development data and validates on the other 20%. Five different models are fitted; the purpose is evaluation and model selection, not producing five final deployed models.

### Is the held-out fold validation or test data?

The answer depends on what we do with its score:

- If we use the fold scores to choose a model, feature set, threshold, or hyperparameter, the held-out folds are acting as **validation data**.
- A separate **final test set** should remain untouched until all choices are finished.
- Some libraries call each held-out fold a `test` fold or return `test_score`. That is API terminology; during model selection, its methodological role is validation.
- If CV is used only once for a final performance estimate and nothing is tuned from it, the folds can serve as the evaluation data. For tuning and unbiased evaluation together, use **nested cross-validation** or keep a separate test set.

The common workflow is:

```text
All data
├── Final test set: locked until the end
└── Development data
    └── K-fold CV: train/validate repeatedly and choose the model

After selection: refit on all development data → evaluate once on final test data
```

### Why it is useful

- Every development sample is used for both training and validation, but never in both roles in the same fold.
- The result depends less on one random split.
- It is especially valuable when the dataset is small.
- The variation across folds reveals whether the estimate is stable.

### Costs and common mistakes

- Training is approximately `K` times more expensive.
- The fold scores are correlated, so their mean is useful but not magical proof of future performance.
- Preprocessing must be fitted **inside each training fold**. Scaling or feature selection before CV leaks information.
- Use `StratifiedKFold` for imbalanced classification, `GroupKFold` when related samples belong to the same entity, and time-aware splitting for time series.
- A large `K` uses more training data per fold but costs more and can produce a higher-variance estimate. `K=5` or `K=10` is a common practical starting point.

## 2. Confusion matrix: what do True, False, Positive, and Negative mean?

`Positive` and `Negative` name the two classes. `True` and `False` say whether the prediction is correct.

| Actual \ Predicted | Positive | Negative |
|---|---:|---:|
| **Positive** | **TP** — True Positive | **FN** — False Negative |
| **Negative** | **FP** — False Positive | **TN** — True Negative |

- **TP:** predicted positive, actually positive.
- **TN:** predicted negative, actually negative.
- **FP:** predicted positive, actually negative — a false alarm.
- **FN:** predicted negative, actually positive — a missed positive case.

Always state which class is positive and which axis is actual or predicted. Otherwise, the same four numbers can be interpreted incorrectly.

## 3. A concrete medical-screening example

Suppose 200 people take a disease-screening test. The positive class means **has the disease**.

- `TP = 40`: sick people correctly detected.
- `FP = 10`: healthy people incorrectly flagged.
- `FN = 20`: sick people incorrectly cleared.
- `TN = 130`: healthy people correctly cleared.

| Metric | Formula | Result | Question answered |
|---|---|---:|---|
| Accuracy | $(TP+TN)/N$ | 85.0% | How often is the classifier correct overall? |
| Precision | $TP/(TP+FP)$ | 80.0% | When it predicts disease, how often is it right? |
| Recall / Sensitivity | $TP/(TP+FN)$ | 66.7% | Of all sick people, how many did it find? |
| Specificity | $TN/(TN+FP)$ | 92.9% | Of all healthy people, how many did it clear? |
| False-positive rate | $FP/(FP+TN)$ | 7.1% | What fraction of healthy people were falsely flagged? |
| False-negative rate | $FN/(FN+TP)$ | 33.3% | What fraction of sick people were missed? |
| F1 score | $2PR/(P+R)$ | 72.7% | How strong is the Precision–Recall balance? |
| MCC | formula below | 0.63 | How correlated are predictions and actual labels? |

Which metric matters depends on the cost of the error:

- Prefer **high Recall** when missing a positive case is dangerous, such as initial disease screening.
- Prefer **high Precision** when acting on a false alarm is expensive, such as an invasive follow-up treatment.
- Use **F1** when both missed positives and false alarms matter and one positive class is the focus.
- Accuracy alone can hide failure on a minority class.

## 4. F1, label swapping, and Macro-F1

Precision, Recall, and binary F1 are asymmetric: their values depend on which class is called positive. This is not about `True` versus `False`; it is about the chosen `Positive` class.

In the example above:

$$F1_{positive}=0.727$$

If the class labels are swapped and the healthy class is treated as positive:

$$F1_{negative}=\frac{2TN}{2TN+FP+FN}=0.897$$

A symmetric summary for binary classification is the **Macro-F1**:

$$Macro\text{-}F1=\frac{F1_{positive}+F1_{negative}}{2}=0.812$$

Therefore, “swap the classes, calculate F1 again, and average” is exactly the binary version of Macro-F1. Do not confuse it with **weighted F1**, which weights each class by its number of samples and can be dominated by the majority class.

## 5. Matthews correlation coefficient (MCC)

MCC uses all four confusion-matrix cells:

$$
MCC=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

Its range is:

- `+1`: perfect agreement;
- `0`: no linear association between prediction and truth;
- `-1`: perfect opposite prediction.

For the medical example, `MCC ≈ 0.63`.

MCC is **invariant when both actual and predicted positive/negative labels are swapped**. That makes it label-symmetric, unlike binary Precision, Recall, and F1. It is also useful for imbalanced data because all four cells affect the result. The precise statement is label symmetry and balanced use of the confusion matrix—not that MCC is immune to every possible dataset or sampling bias.

## 6. KNN versus a decision tree

| Aspect | KNN | Decision tree |
|---|---|---|
| Learning style | Lazy, instance-based learning | Eager, model-based learning |
| During training | Mostly stores the training examples | Searches for feature/threshold splits |
| What is learned | No explicit global abstraction | A hierarchy of rules and regions |
| Prediction | Searches for nearby stored examples | Follows a short path from root to leaf |
| Main cost | Prediction-time distance search | Training-time split search |

KNN is therefore closer to **search**: it postpones most work until a query arrives. A decision tree does much more work in advance. It tries to create an abstraction of the data by partitioning the feature space into increasingly pure regions. The tree is not “understanding” in a human sense, but it does build an explicit reusable model.

## 7. Impurity in classification trees

At a node `A`, let $p_k$ be the proportion of samples belonging to class $k$. A pure node contains only one class and has zero impurity. A mixed node has higher impurity.

### Entropy

$$
H(A)=-\sum_k p_k\log_2 p_k
$$

For a binary node, entropy ranges from `0` for a pure node to `1 bit` for a 50/50 mixture.

#### Why does entropy contain a logarithm?

Claude Shannon formalized information entropy in his 1948 paper *A Mathematical Theory of Communication*. The information carried by an event with probability $p$ is:

$$I(p)=-\log_2 p$$

The logarithm has a crucial property: independent probabilities multiply, but their information should add.

$$
I(pq)=-\log_2(pq)=-\log_2p-\log_2q=I(p)+I(q)
$$

So two independent observations contribute additive information. Rare events are more surprising and carry more information; certain events carry none because $-\log_2(1)=0$. Entropy is the expected information:

$$H=\sum_k p_k I(p_k)=-\sum_k p_k\log_2p_k$$

The minus sign keeps the result non-negative because $\log p\leq 0$ for $0<p\leq1$. Base 2 measures information in **bits**.

### Gini impurity

$$
G(A)=1-\sum_k p_k^2=\sum_k p_k(1-p_k)
$$

Gini impurity has a direct probability interpretation. Draw two class labels independently from the node's class distribution:

- the probability that both draws are class $k$ is $p_k\times p_k=p_k^2$;
- the probability that both labels match is $\sum_k p_k^2$;
- therefore, the probability that they differ is $1-\sum_k p_k^2$.

This explains the square: it comes from **two independent draws of the same class**, not from an arbitrary preference for squared numbers. For a binary node, Gini ranges from `0` to `0.5`, with the maximum at a 50/50 mixture.

## 8. How a tree chooses a split

For a candidate question `q`, compare the impurity of the parent node `A` with the size-weighted impurity of its children $A_j$:

$$
Gain(A,q)=Impurity(A)-\sum_j\frac{|A_j|}{|A|}Impurity(A_j)
$$

- With entropy, this is **Information Gain**.
- With Gini, this is **Gini impurity reduction**.
- The child impurities are weighted by child size so a tiny pure child cannot hide a large impure child.
- The best candidate split has the largest impurity reduction.

For a 50/50 binary parent:

- `Entropy = 1`, `Gini = 0.5`.
- A perfect split creates two pure children: weighted child impurity is `0`, so gain is maximal.
- A useless split leaves both children 50/50: weighted child impurity equals the parent impurity, so gain is `0`.

Entropy and Gini usually rank splits similarly. Gini is algebraically simpler because it avoids logarithms; entropy has a direct information-theoretic interpretation and is somewhat more sensitive to changes in small class probabilities.

## Exam-ready checklist

- A CV fold used for tuning is validation data, even if a library calls it `test_score`.
- Keep a separate final test set, or use nested CV when both tuning and unbiased evaluation are required.
- Fit every data-dependent preprocessing step inside each training fold.
- Positive/Negative identify classes; True/False identify correct/incorrect predictions.
- Precision asks about predicted positives; Recall asks about actual positives.
- Binary F1 changes when the positive class changes; Macro-F1 averages class-wise F1 scores.
- MCC uses TP, TN, FP, and FN and is symmetric to swapping the two class labels.
- KNN searches stored examples; a decision tree learns reusable split rules.
- A good tree split maximizes parent impurity minus weighted child impurity.
- Shannon's logarithm converts multiplied probabilities into additive information.
- The square in Gini comes from the probability of two independent draws matching.

## References

- [Scikit-learn: Cross-validation and the role of the final test set](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn: KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [Scikit-learn: Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Scikit-learn: Precision and Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Scikit-learn: F1 and other evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn: Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
- [Scikit-learn: Decision trees and impurity criteria](https://scikit-learn.org/stable/modules/tree.html)
- [Claude E. Shannon, “A Mathematical Theory of Communication,” 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- [B. W. Matthews, original 1975 paper associated with MCC](https://doi.org/10.1016/0005-2795(75)90109-9)
