

# Data Analysis & Data-driven Decisions

This data set contains of 10.000 observations and 12 columns that tell a story of bank customers. We have information on basic information about them like their age and geographical location as well as numerical data on the credit score for example. It is much more expensive to sign in a new client than keep an already existing one, that's why in this document I'm going to explore what characteristics are linked to customers leaving and try to speculate how we could act against it. Churn prevention and analysis allows banks to develop effective loyalty programs to keep as many customers as possible. In the next document I'm also going to tackle creating a system that predicts the probability of a person leaving.



### Features with examples

| Surname  | CreditScore | Geography | Gender | Age  | Tenure | Balance   | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
| -------- | ----------- | --------- | ------ | ---- | ------ | --------- | ------------- | --------- | -------------- | --------------- | ------ |
| Mitchell | 850         | France    | Female | 39   | 1      | 115046.74 | 4             | 1         | 0              | 190857.79       | 0      |

A little bit of a background behind the features: Tenure refers to the total amount of years the customer spend with the bank. NumOfProducts refers to the amount of bank services the customer is using. IsActiveMembers tells us if the customer has an active membership status.

**In short:**

- We're working with 12 features
- 8 Numerical & 4 categorical 
- 5 Discrete & 4 continuous
- Our target is 'Exited'



## Exploratory Data Analysis

First I'm going to explore the relationship between different features and our target that is the Exit feature. My goal is to see what kind of correlation exists and if there is any kind of insight into why it could be that way. I'm starting with creating a heatmap and later I will proceed to create individual plots between the features.

### Features & The Target

I'm going to plot the continuous values with box-violin plots next to a scatter so the soft changes are more visible. On the other hand, discrete values will be plotted on a circle bar, to better show the % of the observations in relations to the exit.

 

### _1) Correlation_

![Correlation heatmap](/plots/correlation.png)

On the correlation heatmap above we can see that age is in fact, the biggest factor when it comes to closing the bank account, on the second place we have the balance. One feature that has the lowest possible correlation to the Exited feature is IsActiveMember, it might not contribute a lot to our predictions but if you look closer it has a closer relation to the age and it might be something we're looking for. If customers in a certain age group stay or leave because of something that is related to the membership, we could use this information to make changes in the future. For example the base offer of the bank is not satisfying unless a person has a membership, but the membership has requirements, it might limit the amount of people who decide to stay.



![Correlation lines](/plots/correlation-2.png)

![Correlation lines](/plots/correlation-3.png)

I also wanted to plot some correlations in this form, it looks a little bit different and might provide variety of insights. From the first plot we can tell that people who use 3+ products from the bank are more likely to quit which is quite worrying. It could be that these customers have a clear goal in mind on why they want to have an account in a bank and expect certain products to asses them properly, after trying them and not being satisfied enough they could be leaving, ending up looking for another company. In the second plot we can see more correlation between the NumberOfProducts and other features.



### 2) Relationship between features and the target

Now I'm going to take care of creating graphs for correlation between our features and the target that is 'Exited'. I'm starting with the continuous features and then moving into categorical/discrete ones. For the first ones I will be using a box plot with scatter to help us notice small but important changes.

![Box plot for balance](/plots/exited-balance.png)

We're starting with graphing the target against the balance. It makes sense that people with no resources decided to leave the bank, they have no reason to stay but what could leave to them opening an account but leaving is another thing to consider. It also looks like there might be more people with higher balance who decided to leave.

![Box plot for age](/plots/exited-age.png)

From this plot on the other hand, we can tell that younger people are way less likely to leave. There could be some correlation between people being older and having higher balance. It could be also related to the fact that a lot of young people opens their bank account for the first time or not have enough resources to for example put them on a savings account, in that case most banks will offer them very similar things. People with a higher age median, between 40-50 most probably have more specific view on what a bank should provide to them so they might be more inclined to leave ones that are not up their alley. People over 50 leave the least, and it could be because they already found that this bank fits them the best.

![Box plot for salary](/plots/exited-salary.png)

Salary feature is an estimated value and informs about the possible yearly income. It seems like it has barely any relation to the customers leaving. We can only see that there is slightly more customers with higher salary who left.



![Pie chart for tenure](/plots/pie-tenure.png)

Tenure shows information on how long (in years) the customer stayed with the bank. The pie above showcases the amount of people in % with different tenures who left the bank. 10% of the customers left the bank in their first 3 years from creating the account, majority of them (~50%) left after 7 to 9 years. It's a very important information for the bank because since these were long-term customers of theirs, it might mean that something happened in that time-frame that could lead them to leaving.

![Pie chart for number of products](/plots/pie-products.png)

Here we can see in % how many people who used a given number of products left the bank. It seems like there is some correlation between it, most of it is made up from people who used less than 3 products offered by the bank. It could be a random thing, but it could provide some insight too. To check which one is it, I'm going to plot the total number of people who used a number of products with the % of them who left to look for correlation.

![Bar plot for total amount of customers (in %) who use a given amount of product and left](/plots/product-customers.png)

From this bar plot we can extract some very significant information. Percentage of customers who used more than 3 products and left is very high, we can even see that all people who used 4 products left. It might mean that people who use these two have a strong reason to do so and when their expectations are not met, they leave. Thanks to this information the bank can go more in-depth into what the products offer and what kind of characteristics could make the customers exit.

Lastly I'm going to plot a few histograms that could help us display the relationship between the amount of people of a given characteristics and their probability to leave the bank better.

![Histogram for age and exit rate](/plots/hist-age-exit.png)

This histogram shows us a very important information - the older the customer the more likely they are to leave the bank. So far, the age seems to have the biggest role in the equation. 



![Histogram for age and exit rate](/plots/hist-ten-exit.png)

From the tenure histogram we can get a few insights. People who spent between 6 to 8 years with the bank are less likely to leave. People who spend 9, 1 and 0 years are the most likely to leave. This could mean that when people settle down with the bank, it works for them well but at the later point in their life it starts not to. That would connect really well to the plot before where we saw that older customers are more likely to leave. Customers who had 0-1 tenure, I assume that left because they found a different bank that fits their needs better.



![Histogram for credit score and exit rate](/plots/hist-creditscore.png)

The last histogram I created is the one we see above - with the credit score. There is a few important takeaways. People who have a score lower than 470 are way more likely to leave than those above that number. There is also a few very big spikes later, in the middle. It seems like people who have ~620, 750 and 550 credit score are the ones most likely to leave.



### 3) Correlations between features

Now I'm going to plot the features against each other and see if I can extract some other useful information that could possibly help with solving the business problems.

![Box plot for balance against tenure](/plots/bal-ten.png)

It seems like the balance in the moment of saving the observation doesn't relate a lot to the time people spend with the bank. There is a few moments where we have outliers higher and lower than average. Between 2 and 4 years we can clearly see a spike in the balance. We can also see that people with less time spend with the bank have a lower balance.

![Box plots for geography against age with respect to tenure](/plots/age-geo-tenure.png)

On this box plot we can see the relationship between where the customers opened their account and their age. Bar plots are coloured respectively to the tenure. From this plot it's easy to tell that there is way more customers from France with outliers in the older age group. People with 4 years of history with the bank seem to be younger in France and Germany, while we get more older people in the latter who have 2 years of history.

![Box plots for geography against age with respect to having a credit card](/plots/age-geo-card.png)

When it comes to having a credit card, the relationship looks very similar for each country. The only interesting thing to notice there is that there seems to be more customers above the age of 50 in France. In general, there's a even divide between people who have and not have a card.

![Box plots for geography against age with respect to having a credit card](/plots/geo-bal.png)

On this plot on the other hand, we can get some more important insights. It's not exactly related to the churn but still something to keep in mind if the bank were to do more analysis on other features. It seems like the median of the balance is way higher in Germany than in other two countries. In France and Spain less people with a bigger balance are inclined to have a credit card. I assume it's because people with lower income must rely on repaying the credit debt more often.

![Box plots for geography against balance with respect to having a credit card](/plots/geo-bal.png)

## Conclusion

Now that we went trough all the information in the data-set, it's time to analyse the connections between the insights and create a plan on how to approach future decision making related to the issue. First of all, let's start with the biggest factor that we've found - the age. There seems to be a big issue that the bank has with people over the age of 40 and that could be related to a few reasons. People over 40 often have different needs than the younger customers, suddenly it makes a lot of sense for them to invest and save money. Their budget also gets bigger with time. It's a possibility that the company might have an issue with something related to their offer that is either connected to the saving account rate or the fare taken for a certain size of an account.

Second thing that looks quite problematic is the amount of people leaving who have 3 or more products from the bank. It might be that there is some kind of issue in the ecosystem that we are not aware of. If one product links to another but the result of their fusion is not satisfying the customer most likely searches for another bank with a better offer.

The last but not least - credit score. On that one histogram we could clearly see that there might be a slight issue with people who have less than 470 of a credit score to leave the bank. There was also a few times where the amount of customers leaving spiked very significantly in the middle.



## What next?

The data analysis won't mean much if we don't put it into practice. Let's move into creating data-driven actions that could be taken in the future to fix the churn issue. There is three things that the bank should consider doing:

1. Creating new datasets of features on the products offered and how customers interact with them
2. Gathering the data on how the ecosystem works for people above 40
3. Doing a research on what kind of service other banks offer to the group of people who are most likely to exit the bank
4. Starting a survey for customers who are leaving the bank to gather information on the reasons behind the decision

It's a must in this situation to understand how the products affect the customer, if there is a relationship behind the products and the age group and if the goal customers above a certain age have hard time making come true because of the bank's offer. The survey could contain questions on the reason the customer decided to open the account, what was their experience, what kind of products they were interested in and where their service failed. I also think including a rating system for different account features would a good idea since it's easy to process that kind of data and can leave us with some interesting information.



## Predicting the customer leaving

When the bank finds a solution to the question, they will target people who fall under different categories with offers or changes in the infrastructure. In the meantime, there is also some kind of advertisement or offers that could make people who are most likely to leave rethink their decision. Perhaps a special newsletter would do some help too, or a customer satisfaction form that could be filled by a person if they might be leaving in the near future. It would make a big change if the bank could target people who are likely to leave before they do so, and gather data from them that could help preventing the churn. For this task, I'm going to create a model that predicts that kind of behaviour based on the features included in the data set. Although, a bigger amount of features could greatly help the cause, there is still a possibility to get a good accuracy.



_Let's proceed to the next section where [I'm going to create the model.]()_