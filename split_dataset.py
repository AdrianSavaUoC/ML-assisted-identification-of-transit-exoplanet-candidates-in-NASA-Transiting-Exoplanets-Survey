from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_exoplanet_data(planetary_data):
    #split the data
    train, test = train_test_split(planetary_data, test_size=0.30)

    #save to CSV
    train.to_csv("train.csv", index=False)
    test_solution = test['Candidate']
    test = test.drop(['Candidate'], axis=1)
    test_solution.to_csv('test_solution.csv', index=False)
    test.to_csv('test.csv', index=False)

    #train/test split for model training
    X = train.drop(['Candidate'], axis=1)
    Y = train['Candidate']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.30)

    #scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #return everything needed
    return X, Y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

