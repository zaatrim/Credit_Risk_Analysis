# ** Credit_Risk_Analysis

## *Project Overview*
In this Project I will Machine learning to predict Credit Card risk. I will employ different techniques to train and evaluate models with unbalanced classes ; such as imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.


                  
## *Analysis & Results*
### Analysis
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

#### Use Resampling (Undersampling & Oversampling) Models to Predict Credit Risk
1) Load the Data and Perform Basic Data Cleaning
            # Load the data
            file_path = Path('LoanStats_2019Q1.csv.zip')
            df = pd.read_csv(file_path, skiprows=1)[:-2]
            df = df.loc[:, columns].copy()

            # Drop the null columns where all values are null
            df = df.dropna(axis='columns', how='all')

            # Drop the null rows
            df = df.dropna(axis='rows')

            # Remove the `Issued` loan status
            issued_mask = df['loan_status'] != 'Issued'
            df = df.loc[issued_mask]

            # convert interest rate to numerical
            df['int_rate'] = df['int_rate'].str.replace('%', '')
            df['int_rate'] = df['int_rate'].astype('float') / 100


            # Convert the target column values to low_risk and high_risk based on their values
            x = {'Current': 'low_risk'}   
            df = df.replace(x)

            x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
            df = df.replace(x)

            df.reset_index(inplace=True, drop=True)

            df.head()
2) Split the Data into Training and Testing
            # Create our features
            X = df.drop(columns='loan_status')
            X = pd.get_dummies(X)
            # Create our target
            y = df.loc[:, target].copy()
3) Next I will run 3 Oversampling algorithms:

    3.1) Naive Random Oversampling

            # Resample the training data with the RandomOversampler
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=1)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            Counter(y_resampled)
            # Train the Logistic Regression model using the resampled data
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver='lbfgs', random_state=1)
            model.fit(X_resampled, y_resampled)
            # Calculated the balanced accuracy score
            from sklearn.metrics import balanced_accuracy_score
            y_pred = model.predict(X_test)
            balanced_accuracy_score(y_test, y_pred)
            # Display the confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            # Create a DataFrame from the confusion matrix.
            cm_df = pd.DataFrame(
                cm, index=["Actual high_risk", "Actual low_risk"], columns=["Predicted high_risk", "Predicted low_risk"])
            cm_df
            # Print the imbalanced classification report
            from imblearn.metrics import classification_report_imbalanced
            print(classification_report_imbalanced(y_test, y_pred))

    3.2) SMOTE Oversampling
            # Resample the training data with SMOTE
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=1)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            Counter(y_resampled)
            # Train the Logistic Regression model using the resampled data
            model.fit(X_resampled, y_resampled)
            # Calculated the balanced accuracy score
            y_pred = model.predict(X_test)
            balanced_accuracy_score(y_test, y_pred)
            # Display the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Create a DataFrame from the confusion matrix.
            cm_df = pd.DataFrame(
                cm, index=["Actual high_risk", "Actual low_risk"], columns=["Predicted high_risk", "Predicted low_risk"])
            cm_df
            # Print the imbalanced classification report
            print(classification_report_imbalanced(y_test, y_pred))
    3.3) Undersampling:
        
        # Resample the data using the ClusterCentroids resampler
        from imblearn.under_sampling import ClusterCentroids
        cc = ClusterCentroids(random_state=1)
        X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
        Counter(y_resampled)
        # Train the Logistic Regression model using the resampled data
        model.fit(X_resampled, y_resampled)
        # Calculated the balanced accuracy score
        y_pred = model.predict(X_test)
        balanced_accuracy_score(y_test, y_pred)
        # Display the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Create a DataFrame from the confusion matrix.
        cm_df = pd.DataFrame(
            cm, index=["Actual high_risk", "Actual low_risk"], columns=["Predicted high_risk", "Predicted low_risk"])
        cm_df
        # Print the imbalanced classification report
        print(classification_report_imbalanced(y_test, y_pred))


#### Use the SMOTEENN algorithm to Predict Credit Risk
Next I will test a Combination (Over and Under) Sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. I will resample the data using the SMOTEENN algorithm. 

        # Resample the training data with SMOTEENN
        from imblearn.combine import SMOTEENN
        smoteenn = SMOTEENN(random_state=1)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        Counter(y_resampled)
        # Train the Logistic Regression model using the resampled data
        model.fit(X_resampled, y_resampled)
        # Calculated the balanced accuracy score
        y_pred = model.predict(X_test)
        balanced_accuracy_score(y_test, y_pred)
        # Display the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Create a DataFrame from the confusion matrix.
        cm_df = pd.DataFrame(
            cm, index=["Actual high_risk", "Actual low_risk"], columns=["Predicted high_risk", "Predicted low_risk"])
        cm_df
        # Print the imbalanced classification report
        print(classification_report_imbalanced(y_test, y_pred))

#### Use Ensemble Classifiers to Predict Credit Risk
In this section, I will compare two ensemble algorithms to determine which algorithm results in the best performance. I will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier . 
1) for Ensemble Classifiers I will use the same Reading and Basic Data Cleaning code for Resampling. 
next: 
2) Split the Data into Training and Testing
        # Create our features
        X = df.drop(columns='loan_status')
        X = pd.get_dummies(X)
        # Create our target
        y = df.loc[:, target].copy()
        # Check the balance of our target values
        y['loan_status'].value_counts()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
3.1) Balanced Random Forest Classifier
        # Resample the training data with the BalancedRandomForestClassifier
        from imblearn.ensemble import BalancedRandomForestClassifier
        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
        brf.fit(X_train, y_train)
        # Calculated the balanced accuracy score
        y_pred = brf.predict(X_test)
        balanced_accuracy_score(y_test, y_pred)
        # Display the confusion matrix
        confusion_matrix(y_test, y_pred)
        # Print the imbalanced classification report
        print(classification_report_imbalanced(y_test, y_pred))
        # List the features sorted in descending order by feature importance
        importances = brf.feature_importances_
        cols = X.columns
        # Store in a DataFrame
        feature_importances_df = pd.DataFrame({'feature':cols, 'importance': importances})
        feature_importances_df.head(20)
        # Sort the DF
        feature_importances_df.sort_values('importance', ascending=False)

3.2) Easy Ensemble AdaBoost Classifier

        # Train the EasyEnsembleClassifier
        from imblearn.ensemble import EasyEnsembleClassifier
        eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
        eec.fit(X_train, y_train)
        # Calculated the balanced accuracy score
        y_pred = eec.predict(X_test)
        balanced_accuracy_score(y_test, y_pred)
        # Display the confusion matrix
        confusion_matrix(y_test, y_pred)
        # Print the imbalanced classification report
        print(classification_report_imbalanced(y_test, y_pred))



### Results

#### 
#### 
#### 
#### 
#### 

   
## *Summary*
### Advantages

 
