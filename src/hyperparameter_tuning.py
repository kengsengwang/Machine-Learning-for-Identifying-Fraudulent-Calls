from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_model(X_train, y_train):
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 10, 20, 30],
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
