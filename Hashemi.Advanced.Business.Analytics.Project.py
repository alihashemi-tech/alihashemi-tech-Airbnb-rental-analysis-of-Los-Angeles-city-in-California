#!/usr/bin/env python
# coding: utf-8

# ## Airbnb rental analysis of Los Angeles city in California
# 

# In[1]:


import pandas as pd
import numpy as np


# ## Import data

# In[2]:


data = pd.read_csv('Coello-Camarero_Hashemi_Zarzuela_DB3.csv')
print(data.head())


# # In this part, we eliminate some columns that can not play important roles in our analysis. It will reduce the computaional cost.

# In[3]:


data.drop(['listing_url','scrape_id','name','picture_url','host_url','host_thumbnail_url','host_picture_url','host_verifications'], axis=1, inplace=True)
print(data.head())


# ## In this step we need to correct some data type and style in order to use them in our analysis. We need to eliminate $ sign from the price column and also change its format to float

# In[4]:


data['price'] = data['price'].replace({r'\$':''}, regex = True)


# In[5]:


data['price'].tolist()


# ## Now, we change the type of price column.

# In[6]:



data["price"] = [float(str(i).replace(",", "")) for i in data["price"]]


# In[7]:


data.price


# ## In ordert to analze better the bathroom data in our analysis, we need to perform spliting and then getting dummies of bath column

# In[7]:


data[['numberofbath','Bath']] = data.pop('bathrooms_text').str.split(n=1, expand=True)


# In[8]:


data['numberofbath'] = pd.to_numeric(data['numberofbath'],errors='coerce')


# In[10]:


data['numberofbath'].tolist()


# In[11]:


data['Bath'].tolist()


# In[12]:


data.groupby('Bath').size()


# In[13]:


pd.get_dummies(data['Bath'].str.split('|').apply(pd.Series).stack()).sum(level=0)


# ## We need to seperate and get dummies for bath data in order to use them in correlation analysis.

# In[9]:


data = pd.get_dummies(data, prefix=['Bath'], columns=['Bath'])


# In[15]:


data


# In[16]:


data.info()


# In[17]:


# Taking care of missing values


# In[18]:


data.shape


# In[19]:


# We need to find the total number of null for each columns


# In[20]:


data.isnull().sum().tolist()


# # Replacing missing values
# ## As reviews columns are significant elements in our analysis, we need take care of missing values in these columns

# ## We want to find the correlation between all attributes

# In[21]:


corrr = data.corr()


# In[22]:


corrr


# In[23]:


with np.printoptions(edgeitems=50):
    print(corrr)


# ## Based on achieved results, for columns about reviews, 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value', there are just strong correlations between themselves and no strong correlation between them and other columns. For taking care of missing values, one way is use regression technique for finding these null values based on other attributes.

# ## First of all, we create a new dataset for reviews. Then, we prepare data for training and testing.

# In[24]:


msigvol = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']].copy()


# In[25]:


print(msigvol.shape)


# In[26]:


testd = msigvol[msigvol['review_scores_rating'].isnull()]
print(testd.shape)


# In[27]:


msigvol = msigvol.dropna()
print(msigvol.shape)


# In[28]:


y_train = msigvol['review_scores_rating']
X_train = msigvol.drop("review_scores_rating", axis=1)
X_test = testd.drop("review_scores_rating", axis=1)
print(X_train.head())


# In[29]:


X_test.shape


# In[30]:


X_test.dropna()


# ## As we can see, in the testing data which was made by a dataset that all values of 'review_scores_rating' is null (that we want to predict null values and replace the prediction values instead of null values), when we droped NaN, the dataset would be empty. Based on the achieved results, in observations (rows) that we have null values for 'review_scores_rating', we donot have values for 'review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value', too. Then we can not predict null values for for these attributes based on attributes that have meaningfull correlation with each other. In below we are going to see how we can get error in prediction.   

# In[31]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[32]:


y_pred = lr.predict(X_train)


# In[33]:


from sklearn.metrics import r2_score
r2 = r2_score(y_pred, y_train)
r2


# In[34]:


y_pred = lr.predict(X_test)


# ## In this step, we need to  to eliminate all null values which means we are going to lose part of our data.

# In[10]:


data.replace('N/A', np.nan, inplace = True)


# In[11]:


data = data.dropna(axis=0, subset=['review_scores_rating'])


# In[12]:


data = data.dropna(axis=0, subset=['review_scores_accuracy'])


# In[13]:


data = data.dropna(axis=0, subset=['review_scores_cleanliness'])


# In[14]:


data = data.dropna(axis=0, subset=['review_scores_checkin'])


# In[15]:


data = data.dropna(axis=0, subset=['review_scores_communication'])


# In[16]:


data = data.dropna(axis=0, subset=['review_scores_location'])


# In[17]:


data = data.dropna(axis=0, subset=['review_scores_value'])


# In[18]:


data.shape


# ## Now we need to define an element for showing rental demand since we do not have aby specific attributes in our dataset to show this element,
# ## We select 'availability_90' column which shows number of days available for each rental in the next 90 days. we need to do some changes on the values of this attribute to make it ready for showing demanding. We are going to define a new attribite for our dataset as demanding rate.

# In[19]:


data["demanding_rate"] = ((90 - data["availability_90"])/90)*100


# In[44]:


data.head()


# ## We want to get all correlations again.

# In[45]:


dd = data.corr()


# In[46]:


with np.printoptions(edgeitems=50):
    print(dd)


# ## Now we want to check the relation between rating and rental demand

# In[47]:


q2 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','demanding_rate']].copy()


# In[48]:


q2.shape


# In[49]:


q2 = q2.dropna(axis=0, subset=['demanding_rate'])


# In[50]:


q2.shape


# In[51]:


q2.corr()


# ## Based on achieved results, there are week correlation among rental demnad and review_scores_value, review_scores_accuracy, review_scores_rating, review_scores_checkin and having bath in a propoety . And also, there is no meaningful correlation between other fatores here and demanding rate.

# ## If we want to take a look into the all correlations for the demanding rate

# In[52]:


q22 = data[data.columns[1:]].corr()['demanding_rate'][:]


# In[53]:


q22.nlargest(20)


# ## The most meaningful correlations for demanding rate are those with reviews which all of the correlations are weak.

# ## Now we want to check if rents are influenced by the rating system

# In[54]:


q3 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price']].copy()


# In[55]:


q3.corr()


# ## Based on achieved results, there is no correlation between rents amount and rating systems.

# ## In this step, we would like to know which factors impact the pricing.

# In[56]:


q4 = data[data.columns[1:]].corr()['price'][:]


# In[57]:


q4.sort_values()


# In[58]:


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.figure(figsize=(18,8))
q4.plot(kind = 'bar')
plt.ylabel('Correlation')
plt.xlabel('Attributes')
plt.title('Correlation between rental price and other attributes')
plt.show()


# ## Based on the obtained results, there are just moderate correlations between pricing and accomodations, number of bedrooms, number of beds and those places with more than one bath.

# ## In this step, we would like to split and get dumiies of neighborhood and property type since we would like to study them and their influences.

# In[59]:


data = pd.get_dummies(data, prefix=['neighbourhood_cleansed'], columns=['neighbourhood_cleansed'])


# In[60]:


data.info()


# In[95]:


data = pd.get_dummies(data, prefix=['property_type'], columns=['property_type'])


# In[62]:


data


# In[63]:


q5 = data[data.columns[1:]].corr()['price'][:]


# In[64]:


q5.nlargest(10)


# ## As we can see, there is meaningful correlation among price and number of bedrooms, number of bath, property_type_Entire villa (when the type of rental is villa, numbe), accomodation numbers, number of beds and if the property has more than one bath.

# In[65]:


q55 = data[data.columns[1:]].corr()['demanding_rate'][:]


# In[66]:


q55.nlargest(10)


# ## As we can see there is no change for demanding.

# ## For analyzing relationship between price, demanding and amenities, we want to try again with make the dataset smaller.

# In[67]:


q6 = data[['price', 'demanding_rate', 'review_scores_cleanliness', 'amenities']].copy()


# ## If we want to get dummies for amenities:

# In[ ]:


q6 = pd.get_dummies(q6, prefix=['amenities'], columns=['amenities'])


# In[ ]:


q6.shape


# In[ ]:


q66 = q6[q6.columns[1:]].corr()['demanding_rate'][:]


# ## As we can see, there is not feasible to analyze amenities by details since this attribute conyains a huge amount of data and if we want to split it and det dummiies, we need some specific professional software for big data.

# In[ ]:


data.shape


# In[ ]:


data['amenities'].describe()


# ## As a surrogate, we can count number of amenities for each rental and see if the number of amenities is correlated with price of rentals or other desired elements.

# In[20]:


data['number_of_amenities'] = data['amenities'].str.split().str.len()


# In[21]:


data['number_of_amenities']


# In[70]:


data['demanding_rate'].corr(data['number_of_amenities'])


# In[71]:


data['price'].corr(data['number_of_amenities'])


# In[72]:


data.info()


# ## There is no meaningful relationship with number of amenities and price.

# In[73]:


q8 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','number_of_amenities']].copy()


# In[74]:


q8.corr()


# ## Based on the results, there is week correlations between number of amenities and revies which means places with more amenities cah have better reviews.

# ## We want to take a look inside all amenitis 

# In[75]:


data['amenities'].value_counts().idxmax()


# ## As above result shows the most frequent amenities in LA rental houses.

# ## Let's look at the price again

# In[76]:


data['price'].describe()


# ## if we want to find most expensive neighborhood in LA

# ## Note that we need to refress data without getting dummies for neighborhood

# In[98]:


ex = data[(data['price']>225)]


# In[99]:


ex['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[100]:


dff = ex['neighbourhood_cleansed'].value_counts()[:10]
dff.plot(kind = 'bar')
plt.ylabel('Number of Expensive Rental Places in Neighborhood')
plt.xlabel('Neighborhood')
plt.title('Number of Places')
plt.show()


# ## Based on obtained results, these 10 neighborhoods have most expensive rental places in LA.

# In[101]:


ex['neighbourhood_cleansed'].value_counts().idxmax()


# ## And Venice has the most expensive rental place.

# ## Let's look at the deamanding rate again.

# In[102]:


data['demanding_rate'].describe()


# ## if we want to find most demanding neighborhood in LA by Airbnb users.

# In[103]:


ex2 = data[(data['demanding_rate']>90)]


# In[104]:


ex2['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[105]:


dff3 = ex2['neighbourhood_cleansed'].value_counts()[:10]
dff3.plot(kind = 'bar')
plt.ylabel('Counting')
plt.xlabel('Neighborhood')
plt.title('Most Demanding Neighborhoods')
plt.show()


# ## These neighboorhods are the most demanding neighborhood in LA. The amazing point is, most expensive neighborhoods are amost most demanding neighborhood too.

# ## Now we want to know which neighborhoods have best rating for review:

# In[106]:


data['review_scores_rating'].describe()


# In[107]:


ex3 = data[(data['review_scores_rating']>4.9)]


# In[108]:


ex3['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[109]:


dff4 = ex3['neighbourhood_cleansed'].value_counts()[:10]
dff4.plot(kind = 'bar')
plt.ylabel('Counting')
plt.xlabel('Neighborhood')
plt.title('Best Neighborhoods Based on Reviw Rating')
plt.show()


# ## In this step, we would like to use the amenities data in a different way. First, we would like to define a new dataset that all rentals in it have 5 score in reviews and study the pricing and then add amenities to our study.

# In[110]:


ex4 = data[(data['review_scores_rating']>=5)]


# In[111]:


ex4.shape


# In[112]:


data.shape


# In[113]:


ex4['price'].value_counts()[:10].sort_values(ascending=False)


# In[114]:


ex4['price'].value_counts().idxmax()


# ## As we can see the most common rental price among best rental places based on reviews, is 100 dollar. So if we want to provide a rental place in LA for airbnb business, we know the rent price among 50 and 150 dollars are popular values among users of airbnb that want to travel to LA.

# In[115]:


ex4['price'].describe()


# In[116]:


data['price'].describe()


# ## As we can see, most expensive rentals are not located among places with best reviews. As we can see, the change is very meaningful after 75% max of rental prices.

# ## Now it is time to study amenities.

# In[117]:


ex4['amenities'].value_counts()[:20].sort_values(ascending=False)


# ## Based on the results, most frequent amenitis in rentals with review score of 5 are: "Free parking on premises", "Shampoo", "Beach essentials", "Rice maker", "Shower gel", "Dishes and silverware", "Bed linens", "TV with standard cable", "Stove", "Oven", "Pocket wifi", "Baking sheet", "Refrigerator", "Dryer", "Portable fans", "Hot water", "Washer", "Microwave", "Hair dryer", "Trash compactor", "Cooking basics", "Pool table", "Wine glasses", "Dishwasher", "Iron", "Outdoor dining area", "Keurig coffee machine", "Smoke alarm", "Keypad", "Barbecue utensils", "BBQ grill", "Outdoor furniture", "Board games", "Kitchen", "Ethernet connection", "Heating", "First aid kit", "Sound system", "Coffee maker", "Cable TV", "Toaster", "Air conditioning", "Extra pillows and blankets", "Backyard", "Essentials", "Pool", "Body soap", "Wifi", "Fire pit", "Luggage dropoff allowed", "Long term stays allowed", "Cleaning products", "Freezer", "Hangers", "Piano", "Indoor fireplace", "Lock on bedroom door", "Dedicated workspace", "Conditioner", "Mini fridge", "Carbon monoxide alarm", "Patio or balcony", "Free street parking", "Fire extinguisher", "Bathtub".

# In[118]:


ex4['amenities'].value_counts().idxmax()


# In[120]:


df = ex4['amenities'].value_counts()
df = df[:20]


# In[121]:


df


# In[122]:


da2 = ex4['amenities'].value_counts()
da2 = da2[:20]


# In[123]:


remove_words = ['allowed']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))

da2['amenities'] = ex4['amenities'].str.replace(pat, '', regex=True)


# In[124]:


da2['amenities'] = da2['amenities'].str.replace('parking','', regex=True).str.strip('parking')


# In[125]:


da2['amenities'] = da2['amenities'].str.replace('water','', regex=True).str.strip('water')


# In[126]:


da2['amenities'] = da2['amenities'].str.replace('Hot','hot-water')


# In[127]:


da2['amenities'] = da2['amenities'].str.replace('Free','free-parking')


# In[128]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk

top_N = 12


txt7 = da2.amenities.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txt7)
word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

print('All frequencies, including STOPWORDS:')
print('=' * 60)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])
print(rslt)
print('=' * 60)

rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N)[3:10],
                    columns=['Word', 'Frequency']).set_index('Word')

matplotlib.style.use('ggplot')

#rslt.plot.bar(rot=0)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)
plt.title('Most Favorite Amenities Based on Reviews Rating')


# ## As we can see, these amenities are the most favorite ones in those rentals that have best reviews (for sure we ignore some words like [ or ]. If someone would like to buys amenities for his rental, these ones can be as best profitable options to buy.

# ## If we want to see which amenities are most favorite ones based on demanding:

# In[129]:


da3 = ex2['amenities'].value_counts()
da3 = da3[:20]


# In[130]:


remove_words = ['allowed']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))

da3['amenities'] = ex2['amenities'].str.replace(pat, '')


# In[131]:


da3['amenities'] = da3['amenities'].str.replace('parking','', regex=True).str.strip('parking')


# In[132]:


da3['amenities'] = da3['amenities'].str.replace('Free','free-parking')


# In[133]:


top_N =12


txtt = da3.amenities.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txtt)
word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

print('All frequencies, including STOPWORDS:')
print('=' * 60)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])
print(rslt)
print('=' * 60)

rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N)[3:9],
                    columns=['Word', 'Frequency']).set_index('Word')

matplotlib.style.use('ggplot')

#rslt.plot.bar(rot=0)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)
plt.title('Most Favorite Amenities Based on Reviews Rating')
da3['amenities'] = da3['amenities'].str.replace('parking','', regex=True).str.strip('parking')


# ## As we can see, these amenities are the most favorite ones in those rentals that have high demanding rates. 

# ## As an application for end user:

# In[135]:


val = input("Enter your neighborhood: ")
AI = data[data['neighbourhood_cleansed'].str.contains(val)]
AI = AI[(AI['demanding_rate']>=100)]
AI = AI[(AI['review_scores_rating']>=5)]
df1 = AI['accommodates'].mean()
print('The most popular accomodation size in this neighborhood: ', round(df1))
df2 = AI['numberofbath'].mean()
print('The best number of bath in this neighborhood: ', round(df2))
df3 = AI['Bath'].value_counts().idxmax()
print('The best style of bath in this neighborhood: ', df3)
df4 = AI['beds'].mean()
print('The best number of beds in this neighborhood: ', round(df4))
## Amenities
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk

daa = AI['amenities'].value_counts()
daa = daa[:40]


remove_words = ['allowed']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))

daa['amenities'] = AI['amenities'].str.replace(pat, '')
daa['amenities'] = daa['amenities'].str.replace('parking','', regex=True).str.strip('parking')
daa['amenities'] = daa['amenities'].str.replace('[','', regex=True).str.strip('[')
daa['amenities'] = daa['amenities'].str.replace(']','', regex=True).str.strip(']')
daa['amenities'] = daa['amenities'].str.replace('Free','free-parking')

top_N =12


txtt = daa.amenities.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txtt)
word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

print('All frequencies, including STOPWORDS:')
print('=' * 60)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])
print(rslt)
print('=' * 60)

rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N)[3:9],
                    columns=['Word', 'Frequency']).set_index('Word')

matplotlib.style.use('ggplot')

#rslt.plot.bar(rot=0)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)
plt.title('Most Favorite Amenities')
####
df5 = AI['price'].mean()
print('The mean of rental price this neighborhood: ', round(df5))
print('The price description in this neighborhood:', AI['price'].describe())


# In[22]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


# # Now AI

# In[23]:


get_ipython().system('pip install xgboost==1.0.1')


# In[24]:


import xgboost


# In[25]:


from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


# In[26]:


val1 = input("Enter your neighborhood: ")
data2 = data
data2 = data2.dropna(axis=0, subset=['accommodates'])
data2 = data2.dropna(axis=0, subset=['bedrooms'])
data2 = data2.dropna(axis=0, subset=['beds'])
data2 = data2.dropna(axis=0, subset=['numberofbath'])
data2 = data2.dropna(axis=0, subset=['number_of_amenities'])
data2 = data2.dropna(axis=0, subset=['price'])
AI2 = data2[data2['neighbourhood_cleansed'].str.contains(val1)]
AI2 = AI2[(AI2['demanding_rate']>=100)]
AI2 = AI2[(AI2['review_scores_rating']>=5)]

x = AI2[['accommodates','bedrooms','beds','numberofbath','number_of_amenities']]
y = AI2[['price']]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test = train_test_split(x, test_size=0.2, random_state=25)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=25)
#print(x)
#print(x.info())
#print(y)
#print(y.info())
#regressor2 = RandomForestRegressor(random_state = 1)
#regressor2.fit(x, y)
model_xb_3 = MultiOutputRegressor(xgb.XGBRegressor(gamma=0,                 
    learning_rate=0.01,
    max_depth=9,
    n_estimators=150,                                                                    
    subsample=0.8,
    num_parallel_tree=1,
    colsample_bylevel=1,
    scale_pos_weight=1,
    random_state=34))
model_xb_3.fit(x_train, y_train)
y_pred_train_xb_3 = model_xb_3.predict(x_train)
y_pred_test_xb_3 = model_xb_3.predict(x_test)
print('R2 score of training set is {}',format(r2_score(y_train, y_pred_train_xb_3)))
print('R2 score of testing set is {}',format(r2_score(y_test, y_pred_test_xb_3)))

val2 = int(input("Enter number of your accommodates: "))
val3 = int(input("Enter number of your bedrooms: "))
val4 = int(input("Enter number of your beds: "))
val5 = int(input("Enter number of your bath: "))
val6 = int(input("Enter number of your amenities: "))


val_final = np.array([[val2, val3, val4, val5, val6]]) 
#print(val_final)
#y_trainP2 = poly_model.predict(val_final)
y_pred_test_xb_3_j = model_xb_3.predict(val_final)
#print(y_trainP2)
print('The best price for your rental is: {}'.format(y_pred_test_xb_3_j))


# In[27]:


val1 = input("Enter your neighborhood: ")
data2 = data
data2 = data2.dropna(axis=0, subset=['accommodates'])
data2 = data2.dropna(axis=0, subset=['bedrooms'])
data2 = data2.dropna(axis=0, subset=['beds'])
data2 = data2.dropna(axis=0, subset=['numberofbath'])
data2 = data2.dropna(axis=0, subset=['number_of_amenities'])
data2 = data2.dropna(axis=0, subset=['price'])
AI2 = data2[data2['neighbourhood_cleansed'].str.contains(val1)]
AI2 = AI2[(AI2['demanding_rate']>=100)]
AI2 = AI2[(AI2['review_scores_rating']>=5)]

x = AI2[['accommodates','bedrooms','beds','numberofbath','number_of_amenities']]
y = AI2[['price']]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test = train_test_split(x, test_size=0.2, random_state=25)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=25)
#print(x)
#print(x.info())
#print(y)
#print(y.info())
#regressor2 = RandomForestRegressor(random_state = 1)
#regressor2.fit(x, y)
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(max_depth = 2, n_estimators = 20, random_state = 1)
regressor2.fit(x_train, y_train.values.ravel())
y_pred2ts = regressor2.predict(x_test)
# Evaluating the Algorithm
y_pred2tr = regressor2.predict(x_train)
print('R2 score of training set is {}',format(r2_score(y_train, y_pred2tr)))
print('R2 score of testing set is {}',format(r2_score(y_test, y_pred2ts)))

val2 = int(input("Enter number of your accommodates: "))
val3 = int(input("Enter number of your bedrooms: "))
val4 = int(input("Enter number of your beds: "))
val5 = int(input("Enter number of your bath: "))
val6 = int(input("Enter number of your amenities: "))


val_final = np.array([[val2, val3, val4, val5, val6]]) 
#print(val_final)
#y_trainP2 = poly_model.predict(val_final)
y_pred2ts_j = regressor2.predict(val_final)
print('The best price for your rental is: {}'.format(y_pred2ts_j))

