# 8CC00Code
This repository contains code files and utilities for the analysis and modeling of molecular data using various techniques. It provides functions for data preprocessing, feature scaling, dimensionality reduction with Principal Component Analysis (PCA), correlation analysis, logistic regression modeling, and more.

## Installation
```bash
python -m pip install rdkit matplotlib scikit-learn seaborn pandas numpy
```

## Usage
Open the notebook `Group_assignment_notebook.ipynb` and start running the cells.

## License
Free to use

## Code Files

### analysis.py

This file contains several functions for analyzing and preprocessing data using PCA (Principal Component Analysis) and correlation analysis, as well as training and testing a logistic regression model.

#### Function: `correlation(descriptors)`
This function takes in a DataFrame of molecular descriptors and calculates the correlation matrix between the descriptors. It identifies pairs of descriptors that have a correlation coefficient higher than 0.95 and returns a list of these highly correlated pairs.

#### Function: `remove_colinear(descriptors, highly_correlated_pairs)`
This function takes in a DataFrame of molecular descriptors and a list of highly correlated descriptor pairs. It removes one descriptor from each correlated pair to eliminate collinearity. The function returns the updated DataFrame with the removed descriptors.

#### Function: `ScaleDescriptors(descriptors)`
This function takes in a DataFrame of molecular descriptors and performs feature scaling using the Min-Max scaler from scikit-learn. It scales the descriptor values to the range [0, 1] and returns the scaled DataFrame.

#### Function: `plot_variance(descriptors, percentage=0.9)`
This function takes in a DataFrame of molecular descriptors and an optional percentage value (default is 0.9). It performs Principal Component Analysis (PCA) on the descriptors and plots the cumulative explained variance ratio against the number of features. The function also adds horizontal and vertical lines to indicate the specified percentage of variance and the corresponding number of components. It returns the number of components required to explain the specified percentage of variance.

#### Function: `plot_loadings(descriptors, labels, num_components)`
This function takes in a DataFrame of molecular descriptors, a DataFrame of labels, and the number of components. It performs PCA on the descriptors and plots the score plot and loading plot. The score plot shows the projection of the samples onto the first three principal components, with different colors indicating different classes. The loading plot shows the contribution of each descriptor to the principal components. The function visualizes the plots using matplotlib and returns nothing.

#### Function: `feature_rankings(descriptors, num_components)`
This function takes in a DataFrame of molecular descriptors and the number of components. It performs PCA on the descriptors and calculates the average absolute loading for each feature. The features are then ranked based on their average loadings in descending order. The function returns a Series with the feature rankings.

### model.py

This file contains functions for training, testing, and predicting using a logistic regression model.

#### Function: `train(X, y, num_components, degrees=[1,2], use_pca=[True, False], penaltyTypes=['l1','l2'], penaltyStrengths=[1,2])`
This function takes in input features `X`, target variable `y`, the number of components for PCA, and optional parameters for degrees, use of PCA, penalty types, and penalty strengths. It performs logistic regression with polynomial features and PCA using different combinations of parameters. The function performs cross-validation to evaluate the models and selects the best model based on the average score. It returns the best-trained model.

#### Function: `test(pipeline, X_test, y_test)`
This function takes in a trained pipeline, test features `X_test`, and test target variable `y_test`. It evaluates the model by predicting the target variable for the test features and calculating the accuracy. The function

 also prints the test accuracy and displays a confusion matrix using seaborn and matplotlib.

#### Function: `predict(pipeline, X_new, y_new)`
This function takes in a trained pipeline, new features `X_new`, and new target variable `y_new`. It predicts the target variable for the new features using the trained model. The function returns a DataFrame containing the predicted target variable values sorted in descending order along with the original target variable values.

To use these functions, import the respective modules in your Python code and call the desired functions with the appropriate arguments.

```python
import analysis
import model
import dataloader

# Example usage
data = pd.read_csv('data.csv')
descriptors = dataloader.get_molecular_descriptors(data)
labels = dataloader.get_labels(data)

# Perform analysis and modeling
highly_correlated_pairs = analysis.correlation(descriptors)
descriptors = analysis.remove_colinear(descriptors, highly_correlated_pairs)
descriptors = analysis.ScaleDescriptors(descriptors)
analysis.plot_variance(descriptors)
analysis.plot_loadings(descriptors, labels)
feature_rankings = analysis.feature_rankings(descriptors)
model = model.train(descriptors, labels, num_components=3)
model.test(X_test, y_test)
predictions = model.predict(X_new, y_new)
```

Make sure to replace `'data.csv'` with the actual path to your data file.