import pandas as pd



def load_file(file_name, **kwargs):
    try:
        #pass an optional additional keyword as arguments (we need it for skiprows)
        return pd.read_csv(file_name, encoding='ISO-8859-1', **kwargs)
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_name} is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: There was an issue parsing the file {file_name}. Check if the file is in a valid CSV format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the file {file_name}: {e}")
        return None


from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def save_predictions(model, X_test, y_test, y_pred, filename):
    #calculate performance metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    #create a DataFrame with predictions and metrics
    output = pd.DataFrame({
        'TIC ID': X_test['TIC ID'].values,
        'prediction': y_pred,
        'F1 Score': [f1] * len(y_test),
        'AUC': [auc] * len(y_test),
        'Precision': [precision] * len(y_test),
        'Recall': [recall] * len(y_test),
    })

    #save file to CSV
    output.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
