from sklearn.externals import joblib

def make_prediction(model_path, X_input):
    model = joblib.load(model_path)
    prediction = model.predict(X_input)
    return prediction
