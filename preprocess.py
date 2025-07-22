import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from file_manage import load_file

def adjust_display():
    #adjust display settings to show all rows and columns in the output window
    pd.set_option('display.max_rows', None)  #to display all rows
    pd.set_option('display.max_columns', None)  #to display all columns
    pd.set_option('display.width', None)  #to prevent splitting the table by dropping columns on a different page





def load_and_preview_data():
    planetary_raw = load_file('exofop_tess_tois.csv')
    print('\nOriginal Data preview:')
    print(planetary_raw.head(10))
    return load_file('exofop_tess_tois.csv', skiprows=1)

def check_duplicates_and_missing(planetary_data):
    print('\nDuplicate data: ' + str(planetary_data.duplicated().sum()) + '\n')
    print('\nData frame shape:')
    print(planetary_data.shape)
    print('\nColumns:' + str(planetary_data.columns))
    print('\nStatistical overview: \n' + str(planetary_data.describe()))
    print('\nMissing values:\n')
    column_names = planetary_data.columns
    for column in column_names:
        print(column + ' - ' + str(planetary_data[column].isnull().sum()))
    print(planetary_data.isnull().sum() / len(planetary_data) * 100)

def process_planetary_data(planetary_data):
    print('\nData types:\n')
    print(planetary_data.dtypes.to_string())
    planetary_data['Candidate'] = planetary_data['TFOPWG Disposition'].apply(lambda x: 1 if x in ['PC'] else 0)
    planetary_data['Confirmed'] = planetary_data['TFOPWG Disposition'].apply(lambda x: 2 if x in ['CP', 'KP'] else 1 if x in ['PC'] else 0)
    return planetary_data

def plot_candidate_distribution(planetary_data):
    sns.countplot(x='Candidate', data=planetary_data, width=0.25) 
    plt.xticks(ticks=[0, 1], labels=['Not Candidate', 'Planet Candidate'])
    plt.title('Distribution of Planet Candidates')
    for p in plt.gca().patches:
        plt.text(p.get_x() + p.get_width() / 2., p.get_height(), str(p.get_height()), ha="center", va="bottom")
    plt.show()

    sns.countplot(x='Confirmed', data=planetary_data, width=0.25)
    plt.xticks(ticks=[0, 1, 2], labels=['False candidates', 'Planet Candidates', 'Confirmed Planets'])
    plt.title('Distribution of Confirmed Planets')
    for p in plt.gca().patches:
        plt.text(p.get_x() + p.get_width() / 2., p.get_height(), str(int(p.get_height())), ha="center", va="bottom")
    plt.show()
    
    sns.countplot(x='TFOPWG Disposition', data=planetary_data)
    plt.title('Class Distribution')
    for p in plt.gca().patches:
        plt.text(p.get_x() + p.get_width() / 2., p.get_height(), str(p.get_height()), ha="center", va="bottom")
    plt.show()

def plot_pairplot(planetary_data):
    selected_cols = ['Stellar Radius (R_Sun)', 'Planet Radius (R_Earth)', 'Stellar Teff (K)', 'Period (days)', 
                     'Transit Epoch (BJD)', 'Stellar Metallicity', 'Stellar Mass (M_Sun)', 'Depth (mmag)', 
                     'Planet Eq Temp (K)', 'Predicted Mass (M_Earth)', 'Planet Insolation (Earth flux)']
    
    sns.pairplot(planetary_data[selected_cols])
    plt.show()
    
    selected_cols = ['Stellar Radius (R_Sun)', 'Planet Radius (R_Earth)', 
                     'Stellar Mass (M_Sun)', 'Depth (mmag)', 
                     'Predicted Mass (M_Earth)', 'Planet Insolation (Earth flux)']
    sns.pairplot(planetary_data[selected_cols])
    plt.show()
    
    selected_cols = ['Stellar Radius (R_Sun)', 'Stellar Mass (M_Sun)', 'Planet Radius (R_Earth)', 
                      'Stellar Teff (K)','Predicted Mass (M_Earth)',
                     'Planet Eq Temp (K)', 'Planet Insolation (Earth flux)']
    sns.pairplot(planetary_data[selected_cols])
    plt.show()
    

def plot_correlation_heatmap(planetary_data):
    correlation = planetary_data.select_dtypes(include=[float, int]).corr()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, ax=ax)
    plt.show()

def check_missing_data_correlation(planetary_data):
    numeric_cols = planetary_data.select_dtypes(include=[np.number])
    missing_corr = numeric_cols.isnull().astype(int).corrwith(planetary_data['Candidate'])
    missing_corr = missing_corr.dropna()
    print(missing_corr.sort_values(ascending=False))

def drop_unnecessary_columns(planetary_data):
    columns_to_drop = ['CTOI', 'Sectors', 'Comments', 'Date TOI Alerted (by TESS Project)', 
                       'Date TOI Updated (by TESS Project)', 'Date Modified (by ExoFOP-TESS)',
                       'Source', 'Detection', 'TESS Disposition', 'TFOPWG Disposition', 
                       'PM RA (mas/yr)', 'PM Dec (mas/yr)', 
                       # 'Transit Epoch (BJD)',
                       # 'Period (days)', 'Duration (hours)', 'Depth (mmag)', 'Depth (ppm)',
                       # 'Planet Radius (R_Earth)', 'Planet Insolation (Earth flux)',
                       # 'Planet Eq Temp (K)', 'Planet SNR', 'Predicted Mass (M_Earth)', 
                       #droped after analysing the feature importance
                       'Pipeline Signal ID', 'Time Series Observations',
                       'SG3 priority', 'Imaging Observations', 'TESS mag', 'Master priority', 'Confirmed'
                       ]
    planetary_data = planetary_data.drop(columns=columns_to_drop)
    planetary_data.dropna(inplace=True)
    return planetary_data
