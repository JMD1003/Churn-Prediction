from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    
    preds = model.predict(x_test)
    print("Classisfication Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))