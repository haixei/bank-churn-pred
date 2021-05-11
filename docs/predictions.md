# Creating predictions model

In this document I will walk you trough my process of creating a model. I already explored some of the features but there is still a few things to check like sanity of the information or null values. I'm also going to process the data and then finally, create the model. For this purpose I will use LightGBM which is a highly effective implementation of gradient boosting, optimized for a lower training time.

## Processing the features

First I'm going to take care of transforming the values before plotting them, we'll take care of anything that might be an issue to our model as well as encode categorical values.

### 1) Null values and sanity of the information

```Python
# Null values by feature in percentage
CreditScore        0.0
Geography          0.0
Gender             0.0
Age                0.0
Tenure             0.0
Balance            0.0
NumOfProducts      0.0
HasCrCard          0.0
IsActiveMember     0.0
EstimatedSalary    0.0
Exited             0.0
```

When it comes to null values, the data set seems to be well cleaned already, but just to be sure I'm also going to check if all the data is sane. For balance, estimated salary, credit score, tenure and number of products, all values should be above 0. For the binary features, there should not be any information that is not 1 or 0.

```Python
Column Balance is sane: True
Column CreditScore is sane: True
Column EstimatedSalary is sane: True
Column Tenure is sane: True
Column NumOfProducts is sane: True
Column Exited is sane: True
Column Gender is sane: True
Column HasCrCard is sane: True
Column IsActiveMember is sane: True
```



### 2) Exploring skewness and normalizing features

Now I'm going to explore the skewness of the numerical features of our data set and normalize them. For this purpose I'm going to use box-cox since all of our values are positive.

```
Skewness:
CreditScore       -0.071607
Age                1.011320
Tenure             0.010991
Balance           -0.141109
NumOfProducts      0.745568
HasCrCard         -0.901812
IsActiveMember    -0.060437
EstimatedSalary    0.002085
Exited             1.471611
```



