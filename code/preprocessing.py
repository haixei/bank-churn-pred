from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from eda import data
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.metrics import classification_report, plot_roc_curve, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as pltx
import pandas as pd

# Explore the information about the data set
print(data.info())
print(data.shape)

# Null values by feature in percentage
print(round(data.isnull().mean() * 100, 3))

# Check the sanity
sanity = True
positive_cols = ['Balance', 'CreditScore', 'EstimatedSalary', 'Tenure', 'NumOfProducts']
binary_cols = ['Exited', 'Gender', 'HasCrCard', 'IsActiveMember']

# Loop trough the columns
for col in positive_cols:
    sanity = data[data[col] < 0].shape[0] == 0
    print('Column ' + col + ' is sane: ' + str(sanity))

for col in binary_cols:
    sanity = len(data[col].value_counts().values) == 2
    print('Column ' + col + ' is sane: ' + str(sanity))

# Remove outliers
def findThresholds(data, col, low_quantile=0.10, up_quantile=0.90):
    q_low = data[col].quantile(low_quantile)
    q_high = data[col].quantile(up_quantile)

    # Get the range
    range = q_high - q_low
    high_limit = q_high + 1.5 * range
    low_limit = q_low - 1.5 * range

    return low_limit, high_limit

# Are there any outliers in the variables
def has_outliers(data, numeric_columns):
    variable_names = []
    for col in numeric_columns:
        low_limit, high_limit = findThresholds(data, col)
        # Check every column but NumOfProducts, it has only distrete values
        if data[(data[col] > high_limit) | (data[col] < low_limit)].any(axis=None) and col != 'NumOfProducts':
            number_of_outliers = data[(data[col] > high_limit) | (data[col] < low_limit)].shape[0]
            if number_of_outliers > 10:
                variable_names.append(col)

    if len(variable_names) > 0:
        print(variable_names)
    else:
        print('No outliers found.')

# Get numeric columns names
num_col_names = data.select_dtypes(include=['int64', 'float64']).columns.values
has_outliers(data, num_col_names)

# Create the pipeline
# Extract the numerical and categorical features into variables
y = data['Exited']
X = data.drop(['Exited'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

numerical_transformer = PowerTransformer(method='yeo-johnson')
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

# Create the model
model = LGBMClassifier(
        max_bin=250,
        boosting_type='dart',
        max_depth=10
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Set up grid search
param_grid = {
    'model__n_estimators': [100, 300],
    'model__num_leaves': [30, 150],
    'model__learning_rate': [0.01, 0.001]
}

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=24)
gs = GridSearchCV(pipeline, param_grid=param_grid, cv=cv)

# Fit the model
print('Training the model..')
lgbm = gs.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

# Get the report on accuracy and AUC
print(classification_report(y_test, y_pred, digits=2))

# Plot the AUC
y_pred_proba = lgbm.predict_proba(X_test)
plot_roc_curve(lgbm, X_test, y_test)
# >> plt.show()

# Get the accuracy score
mean_score = lgbm.cv_results_['mean_test_score']
print('AUC: ', accuracy_score(y_test, y_pred))
print('Mean CV test score: ', mean_score)
print('Best estimator: ', lgbm.best_params_)

# Plot the cv scores
mean_labels = []
for i in range(len(mean_score)):
    mean_labels.append(i+1)

mean_fig = pltx.line(x=mean_labels, y=mean_score, title=f'Mean test score: {round(mean_score.mean(), 3)}')
# >> mean_fig.show()

# Plot feature importance
feat_index = X_train.loc[:, X_train.columns]
feat_imp = pd.Series(lgbm.best_estimator_.named_steps['model'].feature_importances_[:-8],
                     index=feat_index.columns[:-5]).sort_values(ascending=False)

feat_imp_fig = pltx.bar(x=feat_imp, y=feat_imp.index, title='Top 5 features (importance)', color_discrete_sequence=['#764bcc'])
feat_imp_fig.show()