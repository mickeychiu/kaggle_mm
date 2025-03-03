from autogluon.core.metrics import make_scorer
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score

# Define custom Brier Score function for AutoGluon
ag_accuracy_scorer = make_scorer(name='accuracy',
                                 score_func=accuracy_score,
                                 optimum=1,
                                 greater_is_better=True)

ag_brier_score = make_scorer(name='brier_score',
                                 score_func=brier_score_loss,
                                 optimum=0,
                                 greater_is_better=False)


