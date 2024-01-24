A simple machine learning project on a regression problem. 


#### DataSet 
The datasets used here for predicting the cost of healthcare insurance is from the following Kaggle URL :

https://www.kaggle.com/datasets/akshayhedau/healthcare-costs-prediction-dataset

Hospitalisation Details:

- Customer ID
- year
- month
- date
- children: number of children the customer has
- charges: cost of insurance 
- Hospital tier: the tier of hospital ward that the plan provides for
- City tier
- State ID

Medical Examinations:

- Customer ID
- BMI
- HB1AC: Hemoglobin A1c, a measure of the average blood glucose (sugar) levels over the past 2 to 3 months
- Heart Issues: yes/no
- Any Transplants: yes/no
- Cancer history: yes/no
- NumberOfMajorSurgeries
- smoker: yes/no



#### Conclusions from EDA:

- A majority of the sample does not have children, and the distribution is right skewed (as expected for such a variable). The number of children the customer has is statistically significant in determing the insurace cost.
- Hopistal tier, being from a diffrent state, being a smoker, having heart issue, having transplants and having different numebrs of prior major surgeries is statistically significant in determing the insurace cost.
- City tier, having a history of cancer is not statistically significant in determining insurance cost. 
- There is a positive correlation of BMI and Age to the insurance cost across the two categories of whether or not the customer is a smoker.
- HBA1C does not appear to have any correlation with insurance charges.
- Based on the correlation matrix, in order, whether or not a customer is a smoker, the hospital tier and BMI are the most significant factors influencing insurance cost. 


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, train.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into the script data_preprocessing.py, where features are processed, including removal of missing values. The script outputs the processed dataset.
3. Upon splitting into train and test sets, the data is fed into algo.py containing machine learning algorithms which evaluates the models by giving an R-squared score and plotting a graph of actual vs predicted y-values for some visualization. A band around this diagonal line representing the standard deviation of residuals is shown.




#### Choice of models and evaluations

- This is a regression problem and a random forest classifier and a neural network are used here. The loss function is optimized based on the mean-squared error.

- The choices and evaluations of the above mentioned models are as follows:

1. random forest:
    By building multiple decision trees, and with each tree considering a random subset of features at each split, this helps to reduce overfitting and improving generalization by making it less sensitive to noise and outliers in individual features. Another benefit is also that we can extract the most important features influencing the regression prediction.
    
    The R-squared score is 0.899, demonstrating that the RF algorithm performs quite well for this regression task. Nonetheless, about 12% of the points lie outside one standard deviation of residuals around the diagonal line of actual vs predicted y-values. Further investigation into these points can be done to potentially improve the performance of this model.
    
    
    
2. neural network
    Neural networks are able to model non-linear and complex relationships between input features and the target output. Feature importance is not straightforward, however. A dropout of 0.2 was used between layers, and an Adam optimizer was used. 
    
    The R-squared score is around 0.83 (a seed for initialization was not set), which is decent but does not perform as well as the random forest model above. Approximately 18% of points lie outside one standard deviation of residuals about the diagonal line of actual vs predicted y-values, which is quite significant. Further hyperparameter tuning and investigation into these points might improve model performance.
    
    
#### Feature Importance
The random forest model is able to give a clear interpretation of (relative) feature importance and we will examine it. Top few important features are:

1. whether the person is a smoker
2. BMI
3. Age
4. Hospital tier
5. number of children the customer has
6. HBA1C measure

These agrees with the findings from EDA.



