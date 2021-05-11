import pandas as pd
import numpy as np
import plotly.express as pltx
import plotly.figure_factory as ff

# Load in the data
data = pd.read_csv('../data/churn.csv')
data = data.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)
print(data.head())
print(data.describe())

# Correlation between the features
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Purpor')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()

def getDiscrete(age):
    print(age)
    if age < 29:
        return 18
    elif age > 29 and age <= 39:
        return 30
    elif age > 39 and age <= 49:
        return 40
    elif age > 49 and age <= 69:
        return 50
    else: return 60

data['AgeMin'] = data.apply(lambda row: getDiscrete(row.Age), axis=1)
print(data['AgeMin'])

# Singular features against the target
corr_lines_fig = pltx.parallel_categories(data.head(3000), color="AgeMin", color_continuous_scale='RdPu')
corr_lines_fig.show()

# Drop the AgeMin column, it won't be useful later
data = data.drop(['AgeMin'], axis=1)

exited_fig = pltx.violin(data, y="Balance", x="Exited", color="Exited", box=True, points="all", color_discrete_sequence=['#ff4db5', '#b5abff'])
# >> exited_fig.show()

# Create circle bar plot to show the discrete values
data_exited = data[data['Exited'] == 1]
pie_fig = pltx.pie(data_exited, values='NumOfProducts', names='NumOfProducts', title='Number Of Products', color_discrete_sequence=pltx.colors.sequential.Purpor)
# >> pie_fig.show()

# Plot the % of people who left and used a given amount of products offered by the bank
total_prod = data['NumOfProducts'].value_counts()
exited_prod = data[data['Exited'] == 1]['NumOfProducts'].value_counts()
left_in_perc = {'products': [], 'perc_who_left': []}

for i in range(len(total_prod.values)):
    perc = round((exited_prod.values[i]/total_prod.values[i]) * 100, 1)
    left_in_perc['products'].append(i + 1)
    left_in_perc['perc_who_left'].append(perc)

perc_fig = pltx.bar(left_in_perc, x='products', y='perc_who_left', color_discrete_sequence=['#764bcc'])
# >> perc_fig.show()

# Creating plots for relationships between features
corr_other_box_fig = pltx.box(data, y="Balance", x="Geography", color="HasCrCard", notched=True, color_discrete_sequence=['#fd80ff', '#6f4fff'])
# >> corr_other_box_fig.show()

# Showcase the distribution of age and exited
has_card_yes = data[data['Exited'] == 1]['CreditScore']
has_card_no = data[data['Exited'] == 0]['CreditScore']
hist_data = [has_card_yes, has_card_no]
hist_labels = ['Exited', 'Did not exit']
hist_fig = ff.create_distplot(hist_data, hist_labels, colors=['#fd80ff', '#6f4fff'],
                         bin_size=.2, show_rug=False)

hist_fig.update_layout(title_text='CreditScore histogram')
# >> hist_fig.show()