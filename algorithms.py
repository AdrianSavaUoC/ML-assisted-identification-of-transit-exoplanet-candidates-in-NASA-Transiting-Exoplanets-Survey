import pandas as pd

#pre-process
from sklearn.model_selection import GridSearchCV

#sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

#own helper functions
from evaluation import evaluation_metrics, show_result, cross_validation, features_proba
from preprocess import adjust_display, load_and_preview_data, check_duplicates_and_missing, process_planetary_data, plot_candidate_distribution, plot_pairplot, plot_correlation_heatmap, check_missing_data_correlation, drop_unnecessary_columns
from file_manage import save_predictions
from split_dataset import prepare_exoplanet_data


adjust_display()

#data pre-processing and exploration
planetary_data = load_and_preview_data()
check_duplicates_and_missing(planetary_data)
planetary_data = process_planetary_data(planetary_data)


plot_candidate_distribution(planetary_data)
plot_pairplot(planetary_data)
plot_correlation_heatmap(planetary_data)
check_missing_data_correlation(planetary_data)

planetary_data = drop_unnecessary_columns(planetary_data)
print(planetary_data.head(10))
print(planetary_data.columns)
print(planetary_data.shape)


#split the data for training
X, Y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_exoplanet_data(planetary_data)
 
#apply the Logistic regression model
lr = LogisticRegression(C=100, max_iter=500, class_weight='balanced')
#apply the KNN model
knn = KNeighborsClassifier(leaf_size=3, metric='manhattan', weights='distance')
#Decision Tree Classifier
tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
#Random Forest Classifier
forest = RandomForestClassifier(n_estimators=100, criterion='gini', class_weight='balanced', random_state=42)


#Logistic Regression on non scaled data
def run_lr_non_scaled():
    #fit model to the train data
    lr.fit(X_train, y_train)
    #predict
    y_pred = lr.predict(X_test)
    #evaluate
    print('\nLogistic Regression,\nNon-Scaled data' )
    evaluation_metrics(y_test, y_pred)
    #show Cross-valiadtion scores
    cross_validation(lr, X_train, y_train)
    save_predictions(lr, X_test, y_test, y_pred, 'lr_non_scaled.csv')

    
#Logistic Regression on scaled data
def run_lr_scaled():
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    print('\nLogistic Regression,\nScaled data' )
    evaluation_metrics(y_test, y_pred)
    #show Cross-valiadtion scores
    cross_validation(lr, X_train_scaled, y_train)
    #show feature proba. to influence the prediciton
    features_proba(lr, X_test_scaled, y_test)

    #analyse the importance of the features
    feature_importance = pd.DataFrame(lr.coef_.flatten(), index=X.columns, columns=['Importance'])
    print('\nFeature importance for Logistic Regression algorithm:')
    print(feature_importance.sort_values(by='Importance', ascending=False))
    print(y_train.value_counts(normalize=True))  #check class distribution
    #define the hyperparameters to tune
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']}
    #create GridSearchCV
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    #fit the  data
    grid.fit(X_train_scaled, y_train)
    show_result(grid)
    save_predictions(lr, X_test, y_test, y_pred, 'lr_scaled.csv')

    
def run_lr():
    run_lr_non_scaled()
    run_lr_scaled()
    

#KNeighborsMean
def run_knn():    
    #fit the model with the scaled data
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print('\nKNeighborsClassifier, scaled data:' )
    evaluation_metrics(y_test, y_pred)
    #show Cross-valiadtion scores
    cross_validation(knn, X_train_scaled, y_train,)
    save_predictions(knn, X_test, y_test, y_pred, 'knn_scaled.csv')
    
#Decision Tree
def run_decision_tree():  
    #fit the model
    tree.fit(X_train_scaled, y_train)
    #predict
    y_pred = tree.predict(X_test_scaled)
    #evaluate
    print('\nDecision tree, scaled data:')
    evaluation_metrics(y_test, y_pred)
    #show Cross-valiadtion scores
    cross_validation(tree, X_train_scaled, y_train,)
    save_predictions(tree, X_test, y_test, y_pred, 'dt_scaled.csv')
    
    
#Random Forest
def run_random_forest():
    print('\nRandom Forest Classifier, scaled data:')
    #fit the model with scaled data
    forest.fit(X_train_scaled, y_train)
    #predict
    y_pred = forest.predict(X_test_scaled)
    #evaluate
    evaluation_metrics(y_test, y_pred)    
    #show Cross-valiadtion scores
    cross_validation(forest, X_train_scaled, y_train,)
    features_proba(forest, X_test_scaled, y_test)
    save_predictions(forest, X_test, y_test, y_pred, 'rf_non_scaled.csv')

    #fit the model with non-scaled data
    forest.fit(X_train, y_train)
    #predict
    y_pred = forest.predict(X_test)
    print('\nRandom Forest Classifier, non-scaled data:')
    evaluation_metrics(y_test, y_pred)
    cross_validation(forest, X_train, y_train,)
    save_predictions(forest, X_test, y_test, y_pred, 'rf_scaled.csv')

    
    #feature importance
    importances = pd.Series(forest.feature_importances_, index=X.columns)
    print('\nFeature importance for Random Forest Classifier algorithm:')
    print(importances.sort_values(ascending=False))
    
