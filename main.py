#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:39:38 2022

@author: Adam Fragkiadakis
"""

#%% Load nescessary libraries

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#%% Visualization functions

# Generate a scatter plot
def sactter_plot(x, y):
    plt.scatter(x, y)
    plt.show()
    
# Generate a heatmap
def heatmap_plot(df):
    corrmat = df.corr()
    plt.figure(figsize=(5,5))
    sns.heatmap(corrmat,annot=True)

#%% Pre-processing functions

# Normalize dataframe
def normalize(df):
    res = df.copy()
    for feature_name in df.columns:
        if feature_name == 'painLevels':
            continue
        
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        res[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        
    return res

# Standardize dataframe
def standardize(df):
    res = df.copy()
    for feature_name in df.columns:
        if feature_name == 'painLevels':
            continue
        
        mean_value = df[feature_name].mean()
        std = df[feature_name].std()
        # Z-score using pandas
        res[feature_name] = (df[feature_name] - mean_value) / std
        
    return res

# Find outliers using IQR method
def find_outliers_IQR(df):
    q1 = df.quantile(0.15)
    q3 = df.quantile(0.85)
    IQR = q3 - q1
    
    # Find outliers indices in data frame
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    outliersIndices = []
    for row in outliers.index:
        outliersIndices.append(row)
    
    return outliersIndices

# Find the outliers (put in comments the features that you dont want to look for outliers)
def remove_outliers(df):
    outliersIndices = find_outliers_IQR(df['heart_rate'])
    outliersIndices += find_outliers_IQR(df['IBIs_mean'])
    outliersIndices += find_outliers_IQR(df['RMSSD'])
    outliersIndices += find_outliers_IQR(df['SDNN'])
    outliersIndices += find_outliers_IQR(df['RatioSR'])
    outliersIndices += find_outliers_IQR(df['slope_LR_IBIs'])
    # Remove duplicate indices
    outliersIndices = list(dict.fromkeys(outliersIndices))
    # And remove rows that correspond to the outliers
    df = df.drop(outliersIndices, axis=0)
    
    return df

# Split the data into training and testing datasets
def split_data(feature_labels, df, labelIsAge):
    # 26 samples for validation (13 men and 13 women).
    # First 5 -> low expression, next 21 normal expression.
    # From 21 of normal expression the first 7 have age betweein 20-35
    # the next 7 between 36-50 and the last 7 between 51-65
    validation_samples = [
        '100914_m_39', '101114_w_37',
        '082315_w_60', '083114_w_55',
        '083109_m_60', '072514_m_27',
        '080309_m_29', '112016_m_25',
        '112310_m_20', '092813_w_24',
        '112809_w_23', '112909_w_20',
        '071313_m_41', '101309_m_48',
        '101609_m_36', '091809_w_43',
        '102214_w_36', '102316_w_50',
        '112009_w_43', '101814_m_58',
        '101908_m_61', '102309_m_61',
        '112209_m_51', '112610_w_60',
        '112914_w_51', '120514_w_56'
    ]
    
    # Split validation and training samples
    split_validation = lambda row:row.split('-')[0] in validation_samples
    training_samples = df[~df['ecgFile'].apply(split_validation)].sample(frac=1).reset_index(drop=True)
    testing_samples = df[df['ecgFile'].apply(split_validation)].sample(frac=1).reset_index(drop=True)
    
    # Create train and test sets
    X_train = training_samples[feature_labels]
    X_test = testing_samples[feature_labels]
    if labelIsAge:
        y_train = training_samples['ages']
        y_test = testing_samples['ages']
    else:
        y_train = training_samples['painLevels']
        y_test = testing_samples['painLevels']
    
    return X_train, X_test, y_train, y_test
    

#%% Classification algorythms

# Neural network model
class NeuraNetwork(nn.Module):
    def __init__(self, input_size, number_of_classes):
        super(NeuraNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes),
            # nn.ReLU(),
            # nn.Linear(128, number_of_classes),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Train model
def train_model(X_train, y_train, device, model, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    X_train = X_train.to(device)
    y_train   = y_train.to(device)
    outputs  = model(X_train)

    loss     = criterion(outputs, y_train)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == y_train.data) #acc
    running_loss     += loss.item() * X_train.size(0)    #loss

    ##calculation of gradients and backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    ## caluculation of accuracy & loss     
    epoch_acc  = running_corrects.double() / X_train.size(0)
    epoch_loss = running_loss / X_train.size(0)
    
    training_acc = []
    training_loss = []
    training_acc.append(epoch_acc.cpu().numpy().tolist())
    training_loss.append(epoch_loss)

    if epoch%1000==0:
        print(f'training loss: {epoch_loss:.4f},    acc: {epoch_acc:.4f}')     

    return model, training_acc, training_loss

# Validate model
def valid_model(X_test, y_test, device, model, criterion, epoch):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():        
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        # Model validation          
        outputs  = model(X_test)
        loss     = criterion(outputs, y_test)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == y_test.data) #acc
        running_loss     += loss.item() * X_test.size(0)    #loss
    
    # Model metrics          
    epoch_acc  = running_corrects.double() / X_test.size(0)
    epoch_loss = running_loss / X_test.size(0)   
    
    validation_acc = []
    validation_loss = []
    validation_acc.append(np.squeeze(epoch_acc.cpu().numpy().tolist()))
    validation_loss.append(epoch_loss) 
    
    if epoch%1000==0:
        print(f'validation loss: {epoch_loss:.4f},    acc: {epoch_acc:.4f}')
        print('-' * 50)
    
    return validation_acc, validation_loss, preds

# Print accuracy results
def print_results(alg, test_acc, train_acc, t):
    print(f'--------------{alg}--------------')
    print(f'Testisng Accuracy: {test_acc}')
    print(f'Training Accuracy: {train_acc}')    
    print(f'Time: {time.time()-t}')
    print('----------------------------------------------')
    print('\n')
    
# Neural Network Classification
def nnClassification(number_of_classes, input_size, X_train, X_test, y_train, y_test):
    epochs = 10000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Accuracy and loss
    training_acc    = []
    training_loss   = []
    validation_acc  = []
    validation_loss = []
    
    # DataFrame --> NumPy Array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # numpy --> tensors --> tensors 32bit
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test  = torch.tensor(y_test, dtype=torch.int64)

    model = NeuraNetwork(input_size, number_of_classes).to(device)
    model.float()

    # The optmizer optimize the weights of the model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # The loss function
    criterion = nn.CrossEntropyLoss()
    
    t = time.time()
    for epoch in range(epochs):
        if epoch%1000==0:
            print('Epoch {}/{}'.format(epoch, epochs))
        
        model_trained, training_acc, training_loss = train_model(X_train, y_train, device, model, criterion, optimizer, epoch)
        validation_acc, validation_loss, preds = valid_model(X_test, y_test, device, model, criterion, epoch)
    
    a = validation_acc.index(max(validation_acc))
    print(f'time: {time.time()-t}')
    print('-' * 10)
    print('Acc:', max(validation_acc))
    print('Index:', a) 
    print('Loss:', validation_loss[a])

# LDA Classification
def ldaClassify(feature_labels, X_train, y_train, X_test, y_test):
    lda = LinearDiscriminantAnalysis()
    t = time.time()
    lda.fit(X_train[feature_labels], y_train)
    predict = lda.predict(X_test[feature_labels])
    
    test_accuracy = accuracy_score(y_test, predict)
    train_accuracy = accuracy_score(y_train, lda.predict(X_train[feature_labels]))
    
    print_results('LDA Classification', test_accuracy, train_accuracy, t)

# SVC Classification
def svcClassify(feature_labels, X_train, y_train, X_test, y_test):
    svc = SVC(kernel='linear', random_state=42)
    t = time.time()
    svc.fit(X_train[feature_labels], y_train)
    predict = svc.predict(X_test[feature_labels])
    
    test_accuracy = accuracy_score(y_test, predict)
    train_accuracy = accuracy_score(y_train, svc.predict(X_train[feature_labels]))
    
    print_results('SVM Classification', test_accuracy, train_accuracy, t)
    
# KNN Classification
def knnClassify(feature_labels, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=15)
    t = time.time()
    knn.fit(X_train[feature_labels], y_train)
    predict = knn.predict(X_test[feature_labels])
    
    test_accuracy = accuracy_score(y_test, predict)
    train_accuracy = accuracy_score(y_train, knn.predict(X_train[feature_labels]))
    
    print_results('KNN Classification', test_accuracy, train_accuracy, t)

#%% On file execution...
if __name__ == '__main__':    
    # Load data
    features_df = pd.read_csv('ECGfeatures.csv', delim_whitespace=False, delimiter=';', header=0)
    # Assign in this list only the data that you want the classifiers to use
    # Features: 'heart_rate', 'IBIs_mean', 'RMSSD', 'SDNN', 'RatioSR', 'slope_LR_IBIs'
    feature_labels = ['heart_rate', 'IBIs_mean', 'RMSSD', 'SDNN', 'RatioSR', 'slope_LR_IBIs']
    
    # for fl in feature_labels:
    #     sactter_plot(features_df[fl], features_df['painLevels'])
    # heatmap_plot(features_df[feature_labels])
    
    #%% Normalize data (Uncomment to normalize data)
    # Normalization cannot cope with outliers, remove them before scaling
    # features_df = remove_outliers(features_df)
    # processed_df = normalize(features_df[feature_labels])
    # features_df[feature_labels] = processed_df

    #%% Standardize data (Uncomment to standardize data)
    # Standardization is much less affected by the outliers
    processed_df = standardize(features_df[feature_labels])
    features_df[feature_labels] = processed_df
    features_df = remove_outliers(features_df)
    
    #%% Multiclass classification
    
    ttotal = time.time()
    
    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = split_data(feature_labels, features_df, False)
    # Linear Discrimination Analysis (LDA) Classification
    ldaClassify(feature_labels, X_train, y_train, X_test, y_test)
    # SVC Classification
    svcClassify(feature_labels, X_train, y_train, X_test, y_test)
    # KNN Classification
    knnClassify(feature_labels, X_train, y_train, X_test, y_test)
    # Neural Network Classification
    # input_size = len(feature_labels)
    # output_size = len(features_df['painLevels'].unique())
    # nnClassification(output_size, input_size, X_train, X_test, y_train, y_test)
    
    #%% Binary classification
    
    # Create dataframes based on pain levels
    no_pain = features_df.loc[features_df['painLevels'] == 0]
    pain_level_1 = features_df.loc[features_df['painLevels'] == 1]
    pain_level_2 = features_df.loc[features_df['painLevels'] == 2]
    pain_level_3 = features_df.loc[features_df['painLevels'] == 3]
    pain_level_4 = features_df.loc[features_df['painLevels'] == 4]
    
    # No pain - pain level 1 experiment
    no_pain_pain_1 = pd.concat([no_pain, pain_level_1])
    
    # Pain level 1 - pain level 2 experiment
    # pain_1_pain_2 = pd.concat([pain_level_1, pain_level_2])
    # pain_1_pain_2.loc[pain_1_pain_2['painLevels'] == 1, 'painLevels'] = 0
    # pain_1_pain_2.loc[pain_1_pain_2['painLevels'] == 2, 'painLevels'] = 1
    
    # Pain level 2 - pain level 3 experiment
    # pain_2_pain_3 = pd.concat([pain_level_2, pain_level_3])
    # pain_2_pain_3.loc[pain_2_pain_3['painLevels'] == 2, 'painLevels'] = 0
    # pain_2_pain_3.loc[pain_2_pain_3['painLevels'] == 3, 'painLevels'] = 1
    
    # Pain level 3 - pain level 4 experiment
    # pain_3_pain_4 = pd.concat([pain_level_3, pain_level_4])
    # pain_3_pain_4.loc[pain_3_pain_4['painLevels'] == 3, 'painLevels'] = 0
    # pain_3_pain_4.loc[pain_3_pain_4['painLevels'] == 4, 'painLevels'] = 1
    
    # No pain - pain level 2 experiment
    no_pain_pain_2 = pd.concat([no_pain, pain_level_2])
    no_pain_pain_2.loc[no_pain_pain_2['painLevels'] == 2, 'painLevels'] = 1
    
    # Pain level 1 - pain level 3 experiment
    # pain_1_pain_3 = pd.concat([pain_level_1, pain_level_3])
    # pain_1_pain_3.loc[pain_1_pain_3['painLevels'] == 1, 'painLevels'] = 0
    # pain_1_pain_3.loc[pain_1_pain_3['painLevels'] == 3, 'painLevels'] = 1
    
    # Pain level 2 - pain level 4 experiment
    # pain_2_pain_4 = pd.concat([pain_level_2, pain_level_4])
    # pain_2_pain_4.loc[pain_2_pain_4['painLevels'] == 2, 'painLevels'] = 0
    # pain_2_pain_4.loc[pain_2_pain_4['painLevels'] == 4, 'painLevels'] = 1
    
    # No pain - pain level 3 experiment
    no_pain_pain_3 = pd.concat([no_pain, pain_level_3])
    no_pain_pain_3.loc[no_pain_pain_3['painLevels'] == 3, 'painLevels'] = 1
    
    # Pain level 1 - pain level 4 experiment
    # pain_1_pain_4 = pd.concat([pain_level_1, pain_level_4])
    # pain_1_pain_4.loc[pain_1_pain_4['painLevels'] == 1, 'painLevels'] = 0
    # pain_1_pain_4.loc[pain_1_pain_4['painLevels'] == 4, 'painLevels'] = 1
    
    # No pain - pain level 4 experiment
    no_pain_pain_4 = pd.concat([no_pain, pain_level_4])
    no_pain_pain_4.loc[no_pain_pain_4['painLevels'] == 4, 'painLevels'] = 1
    
    # Binary pain dataframe (The subject feels any intensity of pain or no pain at all)
    binary_pain = features_df.copy()
    binary_pain.loc[binary_pain['painLevels'] > 0, 'painLevels'] = 1
    
    # Put the dataframes in a list to loop through
    dataframes = [
        no_pain_pain_1,
        # pain_1_pain_2,
        # pain_2_pain_3,
        # pain_3_pain_4,
        no_pain_pain_2,
        # pain_1_pain_3,
        # pain_2_pain_4,
        no_pain_pain_3,
        # pain_1_pain_4,
        no_pain_pain_4,
        binary_pain
    ]
    
    # Put the dataframes labels in a list
    dataframe_labels = [
        'No Pain - Pain Level 1',
        # 'Pain Level 1 - Pain Level 2',
        # 'Pain Level 2 - Pain Level 3',
        # 'Pain Level 3 - Pain Level 4',
        'No Pain - Pain Level 2',
        # 'Pain Level 1 - Pain Level 3',
        # 'Pain Level 2 - Pain Level 4',
        'No Pain - Pain Level 3',
        # 'Pain Level 1 - Pain Level 4',
        'No Pain - Pain Level 4',
        'Pain - No Pain'
    ]
    
    # Loop through all the pain combination dataframes and classify data
    for i in range(len(dataframes)):
        print(dataframe_labels[i])
        # Split data into training and testing datasets
        X_train, X_test, y_train, y_test = split_data(feature_labels, dataframes[i], False)
        # Linear Discrimination Analysis (LDA) Classification
        ldaClassify(feature_labels, X_train, y_train, X_test, y_test)
        # SVC Classification
        svcClassify(feature_labels, X_train, y_train, X_test, y_test)
        # KNN Classification
        knnClassify(feature_labels, X_train, y_train, X_test, y_test)
        # Neural Network Classification
        # input_size = len(feature_labels)
        # output_size = len(dataframes[i]['painLevels'].unique())
        # nnClassification(output_size, input_size, X_train, X_test, y_train, y_test)
    
    # Create a dataframe and categortize the ages.
    age_pain = features_df.copy()
    categorize_age = lambda x: 0 if 20 <= x['ages'] <= 35 else 1 if 36 <= x['ages'] <=50 else 2
    age_pain['ages'] = age_pain.apply(categorize_age, axis=1)
    
    # Create testing and training datasets
    feature_labels.append('painLevels')
    X_train, X_test, y_train, y_test = split_data(feature_labels, age_pain, True)
    # Neural Network Classification
    input_size = len(feature_labels)
    output_size = len(dataframes[i]['ages'].unique())
    nnClassification(output_size, input_size, X_train, X_test, y_train, y_test)
    
    print(ttotal)
        
