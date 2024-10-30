from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

banknote_authentication = fetch_ucirepo(id=267)

X = banknote_authentication.data.features.to_numpy()
y = banknote_authentication.data.targets.values.ravel()

logistic_model = LogisticRegression(solver='liblinear')
adaboost_model = AdaBoostClassifier(algorithm='SAMME', n_estimators=50)
binary_cross_entropy_model = LogisticRegression(solver='lbfgs')

common_params = {
    "X": X,
    "y": y,
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)

for ax_idx, estimator in enumerate([logistic_model, adaboost_model, binary_cross_entropy_model]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"{estimator.__class__.__name__}")

plt.show()


