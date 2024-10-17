# - дрібне дерево рішень;
# - глибоке дерево рішень;
# - випадковий ліс на дрібних деревах;
# - випадковий ліс на глибоких деревах.
# передбачити поле "Activity"

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, \
    precision_recall_curve

data = pd.read_csv("bioresponse.csv")

y = data["Activity"]
X = data.drop(columns="Activity")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

small_tree = DecisionTreeClassifier(max_depth=3)
deep_tree = DecisionTreeClassifier(max_depth=None)
rf_small_trees = RandomForestClassifier(n_estimators=100, max_depth=3)
rf_deep_trees = RandomForestClassifier(n_estimators=100, max_depth=None)

small_tree.fit(X_train, y_train)
deep_tree.fit(X_train, y_train)
rf_small_trees.fit(X_train, y_train)
rf_deep_trees.fit(X_train, y_train)

pred_small_tree = small_tree.predict(X_test)
pred_deep_tree = deep_tree.predict(X_test)
pred_rf_small = rf_small_trees.predict(X_test)
pred_rf_deep = rf_deep_trees.predict(X_test)

proba_small_tree = small_tree.predict_proba(X_test)
proba_deep_tree = deep_tree.predict_proba(X_test)
proba_rf_small = rf_small_trees.predict_proba(X_test)
proba_rf_deep = rf_deep_trees.predict_proba(X_test)

metrics = {}

for name, preds, probs in [
    ("Small Tree", pred_small_tree, proba_small_tree),
    ("Deep Tree", pred_deep_tree, proba_deep_tree),
    ("Random Forest (Small Trees)", pred_rf_small, proba_rf_small),
    ("Random Forest (Deep Trees)", pred_rf_deep, proba_rf_deep)
]:
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    logloss = log_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    metrics[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log-Loss": logloss,
        "Confusion Matrix": cm
    }

for model, metrics in metrics.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


# ROC curve
def roc_curve_plot(proba, title):
    pos_probs = proba[:, 1]
    plt.plot([0, 1], [0, 1], linestyle='--', label='Proportion')
    fpr, tpr, _ = roc_curve(y_test, pos_probs)
    plt.plot(fpr, tpr, marker='.', label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()


# Precision recall curve
def precision_recall_curve_plot(proba, title):
    pos_probs = proba[:, 1]
    prop = len(y[y == 1]) / len(y)
    plt.plot([0, 1], [prop, prop], linestyle='--', label='Proportion')
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)
    plt.plot(recall, precision, marker='.', label='PRC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()


# Confusion matrix
def confusion_matrix_plot(pred, title):
    cm = confusion_matrix(y_test, pred)
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot()
    cmd.ax_.set_title('Confusion matrix of ' + title)
    plt.show()


# Move threshold
def move_threshold(proba, title, threshold=0.5):
    y_pred_probs = proba[:, 1]  # Probability for class 1
    y_pred_custom = (y_pred_probs > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    logloss = log_loss(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred_custom)
    moved_threshold_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log-Loss": logloss,
        "Confusion Matrix": cm
    }
    print('---------------------' + title + ' with threshold ' + str(threshold) + '--------------------------')
    for moved_threshold_metric, moved_threshold_metric_value in moved_threshold_metrics.items():
        print(f"  {moved_threshold_metric}: {moved_threshold_metric_value}")

    plt.plot([0, 1], [0, 1], linestyle='--', label='Proportion')
    fpr, tpr, _ = roc_curve(y_test, y_pred_custom)
    plt.plot(fpr, tpr, marker='.', label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' with threshold ' + str(threshold))
    plt.legend()
    plt.show()

    prop = len(y[y == 1]) / len(y)
    plt.plot([0, 1], [prop, prop], linestyle='--', label='Proportion')
    precision, recall, _ = precision_recall_curve(y_test, y_pred_custom)
    plt.plot(recall, precision, marker='.', label='PRC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title + ' with threshold ' + str(threshold))
    plt.legend()
    plt.show()

    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot()
    cmd.ax_.set_title('Confusion matrix of ' + title + ' with threshold ' + str(threshold))
    plt.show()


roc_curve_plot(proba_small_tree, "Small Decision Tree")
precision_recall_curve_plot(proba_small_tree, "Small Decision Tree")
confusion_matrix_plot(pred_small_tree, "Small Decision Tree")
move_threshold(proba_small_tree, "Small Decision Tree", 0.3)

roc_curve_plot(proba_deep_tree, "Deep Decision Tree")
precision_recall_curve_plot(proba_deep_tree, "Deep Decision Tree")
confusion_matrix_plot(pred_deep_tree, "Deep Decision Tree")
move_threshold(proba_deep_tree, "Deep Decision Tree", 0.3)

roc_curve_plot(proba_rf_small, "Small Random Forest")
precision_recall_curve_plot(proba_rf_small, "Small Random Forest")
confusion_matrix_plot(pred_rf_small, "Small Random Forest")
move_threshold(proba_rf_small, "Small Random Forest", 0.3)

roc_curve_plot(proba_rf_deep, "Deep Random Forest")
precision_recall_curve_plot(proba_rf_deep, "Deep Random Forest")
confusion_matrix_plot(pred_rf_deep, "Deep Random Forest")
move_threshold(proba_rf_deep, "Deep Random Forest", 0.3)
