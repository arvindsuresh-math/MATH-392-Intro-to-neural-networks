# MATH 392 Midterm Project

#### Dataset
For this project, you will use the datasets found in `data/classification/presidential_election_popular_vote`

This datasets contains county-level demographic and voting outcome data for 4 presidential elections (2008, 2012, 2016, 2020). 

#### Task
You will use the datasets to build a classification model that predicts the outcome of the presidential election based on demographic data. You will also analyze the model's performance and interpret the results.


#### Deadlines:
1. **Proposal (draft)**: By **Monday, March 31**, write (and submit to D2L) a short proposal that describes your project. It should include the following:
   - *Data description*: A brief description of the dataset you will use (what is the training set, what is the test set, what is the target variable, what are the features, etc.)
   - *Problem statement*: Clearly state the problem you are trying to solve (e.g. "predict the overall outcome of the presidential election based on demographic data", or "predict the outcome of the presidential election in a specific county based on demographic data"). 
   - *Modeling approach*: Summarize the models you will build (including loss functions), including what you take as a "baseline" model (against which you will compare all your other models).
   - *Mathematical formulation*: Explain the mathematical formulation of how you will use the models to solve the problem.
   - *Key performance indicators*: Describe the criteria you will use to evaluate the model's performance (e.g. accuracy, precision, recall, F1 score, etc.).
2. **Exploratory data analysis (draft)**: By **Monday, April 7**, submit a Jupyter notebook that contains the following:
   - *Exploratory data analysis*: Perform exploratory data analysis to understand the data and identify any patterns or trends. This includes visualizing the data, calculating summary statistics, and identifying correlations (and mutual information) between features. You should include a few interesting histograms, scatter plots, box plots, and so on, to illustrate your findings.
   - *Feature engineering*: Carry out appropriate transformations of features to try to better distinguish the classes that your are trying to predict. This may include:
        - *Scaling/normalizing*
        - *Transforming features* (using logs or other transformations)
        - *Creating new features* (e.g. combining existing features, creating polynomial interaction terms, etc.)
   - *Feature selection*: Select the most important features for modeling. This can be done using techniques such as correlation analysis, feature importance scores, or recursive feature elimination.
3. **Modeling (draft)**: By **Wednesday, April 9**, submit a Jupyter notebook that contains the following:
    - *Modeling*: Build and evaluate the models you proposed in your proposal. This includes:
        - *Baseline model*: Build a baseline model using a simple classification algorithm (e.g. guess the majority class)
        - *Advanced models*: Build more advanced models (e.g. logistic/softmax regression, multilayer perceptron, QDA, Gaussian Naive Bayes, etc.) and compare their performance to the baseline model.
        - *Ensemble methods*: Experiment with ensemble methods (e.g. stacking, voter models) to improve the performance of your models.
        - *Hyperparameter tuning*: Use techniques such as grid search or random search to tune the hyperparameters of your models and improve their performance.
    - *Model evaluation*: Evaluate the performance of your models using the key performance indicators you described in your proposal. 
    - *Concluding remarks*: Summarize your findings and discuss the strengths and weaknesses of your models.
4. **Final drafts**: By **Monday, April 14**, submit final versions of your proposal, exploratory data analysis, and modeling notebooks.