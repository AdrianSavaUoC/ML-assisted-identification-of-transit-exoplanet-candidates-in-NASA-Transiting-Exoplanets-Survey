import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics._classification import confusion_matrix
from sklearn.model_selection import cross_val_score

def evaluation_metrics(y_test, y_pred):
    #Accuracy, Recall, F1 Score and Precision metrics
    print('\nEvaluation Metrics:')
    print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))
    print('Recall Score: ' + str(metrics.recall_score(y_test, y_pred)))
    print('F1 score: ' + str(metrics.f1_score(y_test, y_pred)))
    print('Precision: ' + str(metrics.precision_score(y_test, y_pred)) + '\n')
    
    #confusion matrix
    print('C Matrix:')
    print('TN,    FP,    FN,    TP')
    print(str(confusion_matrix(y_test, y_pred).ravel()) + '\n')
    
#show best parameters for GridSearchCV
def show_result(results):
    print('\nBest Parameters: ')
    print(results.best_params_)
    print('\nBest Cross-Validation Score: ')
    print(f"{results.best_score_:.4f}")
    
    print('\nBest Estimator:')
    print(results.best_estimator_)
  
#show the probability of a feature to contribute to prediction  
def plot_roc(y_test, y_proba):

    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()
    print(f'ROC Curve (AUC = {roc_auc:.2f})')
    

def cross_validation(model_name, X_train, y_train,):
    #show Cross-valiadtion scores
    scores = cross_val_score(model_name, X_train, y_train, cv=10, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Mean:", scores.mean(), "Std:", scores.std())
    
def features_proba(model,X_test_scaled, y_test):
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    plot_roc(y_test, y_proba)

    