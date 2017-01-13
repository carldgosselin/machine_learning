# UDACITY - Machine Learning Engineer Nanodegree

## Model Evaluation & Validation Project

### P1: Predicting Boston Housing Prices

**Project Overview**

In this project, you will apply basic machine learning concepts on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home. You will first explore the data to obtain important features and descriptive statistics about the dataset. Next, you will properly split the data into testing and training subsets, and determine a suitable performance metric for this problem. You will then analyze performance graphs for a learning algorithm with varying parameters and training set sizes. This will enable you to pick the optimal model that best generalizes for unseen data. Finally, you will test this optimal model on a new sample and compare the predicted selling price to your statistics.

**Project Highlights**

This project is designed to get you acquainted to working with datasets in Python and applying basic machine learning techniques using NumPy and Scikit-Learn. Before being expected to use many of the available algorithms in the sklearn library, it will be helpful to first practice analyzing and interpreting the performance of your model.

**Things you will learn by completing this project:**

How to use NumPy to investigate the latent features of a dataset.
How to analyze various learning performance plots for variance and bias.
How to determine the best-guess model for predictions from unseen data.
How to evaluate a model’s performance on unseen data using previous data.

###Install

This project requires Python 2.7 and the following Python libraries installed:

* NumPy
* scikit-learn

### Run

In the terminal, navigate to the project folder and run:

`ipython boston_housing_students_v2.py`

After running the command above, the following graph below will appear:

![graph #1] (https://github.com/carldgosselin/predicting-boston-housing-market/blob/master/example/Screen%20Shot%202016-04-17%20at%203.16.09%20PM.png)

Note:  You need to close (X) this graph to view the other graphs.

### Data

The dataset used in this project is included with the scikit-learn library (sklearn.datasets.load_boston). You don't have to download it separately.  The data contains the following attributes for each housing area:

* CRIM: per capita crime rate by town
* ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS: proportion of non-retail business acres per town
* CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* NOX: nitric oxides concentration (parts per 10 million)
* RM: average number of rooms per dwelling
* AGE: proportion of owner-occupied units built prior to 1940
* DIS: weighted distances to five Boston employment centres
* RAD: index of accessibility to radial highways
* TAX: full-value property-tax rate per $10,000
* PTRATIO: pupil-teacher ratio by town
* B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
* LSTAT: % lower status of the population
* MEDV: Median value of owner-occupied homes in $1000's
