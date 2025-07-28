samples = [
    (1, 0.9),  # A
    (0, 0.8),  # B
    (1, 0.7),  # C
    (0, 0.6),  # D
    (1, 0.4),  # E
    (0, 0.1),  # F
]

postives = [score for label, score in samples if label == 1]
negatives = [score for label, score in samples if label == 0]

score_sum = 0
total_pairs = len(postives) * len(negatives)
for pos in postives:
    for neg in negatives:
        if pos>neg:
            score_sum += 1
        elif pos == neg:
            score_sum += 0.5
auc = score_sum / total_pairs
print(f"AUC = {auc:.4f}")

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# True labels and predicted probabilities
y_true = [1, 0, 1, 0, 1, 0]
y_scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# AUC score (method 1)
auc_score = auc(fpr, tpr)

# AUC score (method 2: direct)
roc_auc = roc_auc_score(y_true, y_scores)

# Output
print(f"AUC (using auc)       = {auc_score:.4f}")
print(f"AUC (using roc_auc_score) = {roc_auc:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve using Scikit-learn')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

def auc_trapezoidal(y_true, y_scores):
    combined = list(zip(y_true, y_scores))
    combined.sort(key=lambda x: x[1], reverse=True)

    P = len(y_true)
    N = len(y_true) - P
    if P == 0 or N==0:
        return 0
    TPR = [0.0]
    FPR = [0.0]
    TP = 0
    FP = 0

    for label, _ in combined:
        if label == 1:
            TP += 1
        else:
            FP += 1
        TPR.append(TP/P)
        FPR.append(FP/P)
    auc = 0.0
    for i in range(1, len(TPR)):
        width = FPR[i] - FPR[i-1]
        height = (TPR[i] + TPR[i-1]) / 2
        auc += width * height
    return auc 