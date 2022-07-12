#!/usr/bin/env python
# coding: utf-8

# # Created by  Ali Hashemi
# 
# ## March 2022
# 
# 
# 

# ## Airbnb rental analysis of Los Angeles city in California

# In[ ]:





# ## Questions in this project:
# ## 1- Define a proper parameter which shows demanding rates of rentals.
# ## 2- Are review rating's scores related to the rental demands?
# ## 3- Rentals price are influenced by the review rating system?
# ## 4- Most effective attributes on a rental's price?
# ## 5- What are the most important amenities for obtaining max reviwes rates and max demanding rates.

# ## Main Part: Develope an Artificial Intelligence model for pricing rental to achieve max demand and review rate based on effective parameters and location.

# In[2]:


import pandas as pd
import numpy as np


# In[29]:


get_ipython().system('pip install jupyterthemes')


# In[34]:


get_ipython().system('pip install jupyterthemes')
get_ipython().system('jt -t chesterish')


# In[36]:


from jupyterthemes import get_themes
import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme


# In[57]:


set_nb_theme('onedork')


# ## Import data

# In[3]:


data = pd.read_csv('AirBnB-Project-March2022.csv')


# In[12]:


data.describe()


# ### We have 1,399,831 data points - Unstructured

# In[4]:


data.head()


# In[4]:


data.info()


# # Data Dimension Reduction
# 
# ## Eliminate not necessary columns to reduce the computaional cost.

# In[4]:


data.drop(['listing_url','scrape_id','name','picture_url','host_url','host_thumbnail_url','host_picture_url','host_verifications'], axis=1, inplace=True)
print(data.head())


# ## Improvment of some data type and style. First, eliminate dollar sign from the price column and also change its format to float.

# In[6]:


data['price'] = data['price'].replace({r'\$':''}, regex = True)


# In[7]:


data['price'].tolist()


# ## Change type of price column.

# In[8]:



data["price"] = [float(str(i).replace(",", "")) for i in data["price"]]


# In[7]:


data.price


# ## To analze better the bathroom data, I need to perform spliting and then getting dummies of bath column

# In[9]:


data[['numberofbath','Bath']] = data.pop('bathrooms_text').str.split(n=1, expand=True)


# In[10]:


data['numberofbath'] = pd.to_numeric(data['numberofbath'],errors='coerce')


# In[11]:


data['numberofbath'].tolist()


# In[12]:


data['Bath'].tolist()


# In[15]:


data.groupby('Bath').size()


# In[13]:


pd.get_dummies(data['Bath'].str.split('|').apply(pd.Series).stack()).sum(level=0)


# ## Separating and getring dummies for bath data in order to use them in correlation analysis.

# In[13]:


#data = pd.get_dummies(data, prefix=['Bath'], columns=['Bath'])


# In[13]:


data


# In[19]:


data.info()


# # Taking care of missing values

# In[20]:


data.shape


# ## Total number of null for each columns

# In[14]:


data.isnull().sum().tolist()


# # Replacing missing values
# ## Taking care of missing values in reviews columns

# ## First, get the correlation between all attributes

# In[22]:


corrr = data.corr()


# In[23]:


corrr


# In[24]:


with np.printoptions(edgeitems=50):
    print(corrr)


# ## Initial results:
# ## For columns about reviews, 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', there are just strong correlations between themselves and no strong correlation between them and other columns. 
# 
# ## For taking care of missing values:
# 
# ## Runnung regression technique for finding these null values based on other attributes.

# ## Creating new dataset for reviews. 
# ## Next, preparing data for training and testing.

# In[25]:


msigvol = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']].copy()


# In[26]:


print(msigvol.shape)


# In[27]:


testd = msigvol[msigvol['review_scores_rating'].isnull()]
print(testd.shape)


# In[28]:


msigvol = msigvol.dropna()
print(msigvol.shape)


# In[29]:


y_train = msigvol['review_scores_rating']
X_train = msigvol.drop("review_scores_rating", axis=1)
X_test = testd.drop("review_scores_rating", axis=1)
print(X_train.head())


# In[46]:


X_test.shape


# In[47]:


X_test.dropna()


# ## Results: In the testing data which was made by a dataset that all values of 'review_scores_rating' is null that I want to predict null values and replace the prediction values instead of null values, when I droped NaN, the dataset would be empty. Based on the achieved results, in observations (rows) that I have null values for 'review_scores_rating', we donot have values for 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', too. Then predicting null values for for these attributes based on other attributes with meaningfull correlation with each other are not feasible in this stage. 
# 
# ## Checking the error in prediction.   

# ## Using linear regression technique 

# In[71]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[72]:


from sklearn.metrics import r2_score
y_pred1 = lr.predict(X_train)
r2_1 = r2_score(y_pred1, y_train)
r2_1


# ## Using K nearest neighbors technique

# In[74]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
rmse_val = [] #to store rmse values for different k
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred = model.predict(X_train) #make prediction on test set
    R2_KNN = r2_score(pred, y_train)
    error = sqrt(mean_squared_error(y_train,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('R2 value for k= ' , K , 'is:', R2_KNN)
    print('RMSE value for k= ' , K , 'is:', error)


# ## Stick to KNN with K=1 for prediction missing values

# ## Replacing

# In[77]:


model_replacing = neighbors.KNeighborsRegressor(n_neighbors = 1)
model.fit(X_train, y_train)  #fit the model
miss_values = model.predict(X_train)
data.loc[sigvol.isnull()] = y_pred


# ## Eliminate all null values, (losing part of our data).

# In[15]:


data.replace('N/A', np.nan, inplace = True)


# In[16]:


data = data.dropna(axis=0, subset=['review_scores_rating'])


# In[17]:


data = data.dropna(axis=0, subset=['review_scores_accuracy'])


# In[18]:


data = data.dropna(axis=0, subset=['review_scores_cleanliness'])


# In[19]:


data = data.dropna(axis=0, subset=['review_scores_checkin'])


# In[20]:


data = data.dropna(axis=0, subset=['review_scores_communication'])


# In[21]:


data = data.dropna(axis=0, subset=['review_scores_location'])


# In[22]:


data = data.dropna(axis=0, subset=['review_scores_value'])


# In[20]:


data.shape


# In[34]:


data.head()


# ## Need to get all correlations again.

# In[89]:


dd = data.corr()


# In[90]:


with np.printoptions(edgeitems=50):
    print(dd)


# # Question answer: check the relation between rating and rental demand

# ### Now, Defining an element for showing rental demand.
# ### I select 'availability_90' column which shows number of days available for each rental in the next 90 days. 
# 
# ### It needs some changes on the values to make itself ready for showing demanding. 
# 
# ### Defining a new attribite for the dataset as demanding rate.

# ## Q1 answer

# In[23]:


data["demanding_rate"] = ((90 - data["availability_90"])/90)*100


# In[24]:


q2 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','demanding_rate']].copy()

## Built a new dataset including needed columns to check better correlation


# In[92]:


q2.shape


# In[25]:


q2 = q2.dropna(axis=0, subset=['demanding_rate'])


# In[94]:


q2.shape


# In[95]:


q2.corr()


# ## Q2 answer: Based on achieved results, there are weak correlation among rental demnad and review_scores_value, review_scores_accuracy, review_scores_rating, review_scores_checkin and having bath in a propoety . And also, there is no meaningful correlation between other fatores here and demanding rate.

# ## Now taking a look into the all correlations for the demanding rate

# In[96]:


q22 = data[data.columns[1:]].corr()['demanding_rate'][:]


# In[97]:


q22.nlargest(20)


# ## The most meaningful correlations for demanding rate are review scores value and accuracy which all of the correlations are weak.

# # Now, checking if rents are influenced by the rating system

# In[98]:


q3 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price']].copy()


# In[99]:


q3.corr()


# ## Q3 answer: Based on achieved results, there is no correlation between rent price and rating systems.

# ## Check which factors impact the pricing.

# In[100]:


q4 = data[data.columns[1:]].corr()['price'][:]


# In[104]:


q4 = q4.drop('price')


# In[107]:


q4 = q4.abs()


# In[108]:


q4.sort_values()


# ## Question answer: study impact of attributes on rental price

# ## Visulization

# In[109]:


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


# ## Q4 answer : Based on the obtained results, there are correlations between pricing and accomodations, number of bedrooms, number of beds and those places with more than one bath.

# ## In this step, we would like to split and get dumiies of neighborhood and property type since we would like to study them and their influences.

# In[26]:


#data = pd.get_dummies(data, prefix=['property_type'], columns=['property_type'])


# In[27]:


data


# In[24]:


q5 = data[data.columns[1:]].corr()['price'][:]


# In[25]:


q5.nlargest(10)


# In[26]:


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.figure(figsize=(18,8))
q5.plot(kind = 'bar')
plt.ylabel('Correlation')
plt.xlabel('Attributes')
plt.title('Correlation between rental price and other attributes')
plt.show()


# ## There is meaningful correlation among price and number of bedrooms, number of bath, property_type_Entire villa (when the type of rental is villa), accomodation numbers, number of beds and if the property has more than one bath.

# In[114]:


q55 = data[data.columns[1:]].corr()['demanding_rate'][:]


# In[115]:


q55.nlargest(10)


# ## As we can see there is no change for demanding.

# ## Analyzing relationship between price, demanding and amenities

# In[116]:


q6 = data[['price', 'demanding_rate', 'review_scores_cleanliness', 'amenities']].copy()


# ## If we want to get dummies for amenities:

# In[ ]:


q6 = pd.get_dummies(q6, prefix=['amenities'], columns=['amenities'])


# In[ ]:


q6.shape


# In[ ]:


q66 = q6[q6.columns[1:]].corr()['demanding_rate'][:]


# ## As we can see, there is not feasible to analyze amenities by details since this attribute conyains a huge amount of data and if we want to split it and det dummiies, we need some specific professional software for big data.

# In[117]:


data.shape


# In[23]:


data['amenities'].describe()


# ## As a surrogate, count number of amenities for each rental and see if the number of amenities is correlated with price of rentals or other desired elements.

# In[28]:


data['number_of_amenities'] = data['amenities'].str.split().str.len()


# In[25]:


data['number_of_amenities']


# In[123]:


data['demanding_rate'].corr(data['number_of_amenities'])


# In[124]:


data['price'].corr(data['number_of_amenities'])


# In[125]:


data.info()


# ## There is no meaningful relationship with number of amenities and price.

# In[126]:


q8 = data[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','number_of_amenities']].copy()


# In[127]:


q8.corr()


# ## Based on the results, there is week correlations between number of amenities and reviews which means places with more amenities normally can have better reviews.

# ## Take a look inside all amenitis 

# In[26]:


data['amenities'].value_counts().idxmax()


# ## As above result shows the most frequent amenities in LA rental houses.

# ## Analyzing price 

# In[129]:


data['price'].describe()


# ## Mission: Find the most expensive neighborhood in LA

# ## Note that we need to refress data without getting dummies for neighborhood

# In[27]:


ex = data[(data['price']>225)]
## I used mean vlues of price


# In[28]:


ex['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[34]:


dff = ex['neighbourhood_cleansed'].value_counts()[:10]
dff.plot(kind = 'bar')
plt.ylabel('Number of Expensive Rental Places in Neighborhood')
plt.xlabel('Neighborhood')
plt.title('Number of Places')
plt.show()


# ## Based on obtained results, these 10 neighborhoods have most expensive rental places in LA.

# In[101]:


ex['neighbourhood_cleansed'].value_counts().idxmax()


# ## Venice neghiborhood has most expensive rental places in LA

# ## Looking at the deamanding rate again.

# In[134]:


data['demanding_rate'].describe()


# ## Mission: Find the most demanding neighborhoods in LA by Airbnb users.

# In[40]:


ex2 = data[(data['demanding_rate']>90)]
## demanding rate up to 90%


# In[36]:


ex2['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[138]:


dff3 = ex2['neighbourhood_cleansed'].value_counts()[:10]
dff3.plot(kind = 'bar')
plt.ylabel('Counting')
plt.xlabel('Neighborhood')
plt.title('Most Demanding Neighborhoods')
plt.show()


# ## These neighboorhods are the most demanding neighborhood in LA. 
# 
# ## The  point is, most expensive neighborhoods are the most demanding neighborhood too.

# ## Mission:  Find which neighborhoods have best rating for review:

# In[139]:


data['review_scores_rating'].describe()


# In[140]:


ex3 = data[(data['review_scores_rating']>4.9)]


# In[141]:


ex3['neighbourhood_cleansed'].value_counts()[:10].sort_values(ascending=False)


# In[142]:


dff4 = ex3['neighbourhood_cleansed'].value_counts()[:10]
dff4.plot(kind = 'bar')
plt.ylabel('Counting')
plt.xlabel('Neighborhood')
plt.title('Best Neighborhoods Based on Reviw Rating')
plt.show()


# ## In this step, analyzing amenities data in a different way. 
# 
# ## First, I define a new dataset that all rentals in it have 5 score in reviews 

# In[30]:


ex4 = data[(data['review_scores_rating']>=5)]


# In[144]:


ex4.shape


# In[145]:


data.shape


# In[146]:


ex4['price'].value_counts()[:10].sort_values(ascending=False)


# In[147]:


ex4['price'].value_counts().idxmax()


# ## The most common rental price among best rental places based on reviews, is 100 dollar. So if we want to provide a rental place in LA for airbnb business, we know the rent price among 50 and 150 dollars are popular values among users of airbnb that want to travel to LA.

# In[148]:


ex4['price'].describe()


# In[149]:


data['price'].describe()


# ## Result: The most expensive rentals are not located among places with best reviews. Hint, the change is very meaningful after 75% max of rental prices.

# ## Mission: Study amenities.

# In[150]:


ex4['amenities'].value_counts()[:20].sort_values(ascending=False)


# ### Based on the results, most frequent amenitis in rentals with review score of 5 are: "Free parking on premises", "Shampoo", "Beach essentials", "Rice maker", "Shower gel", "Dishes and silverware", "Bed linens", "TV with standard cable", "Stove", "Oven", "Pocket wifi", "Baking sheet", "Refrigerator", "Dryer", "Portable fans", "Hot water", "Washer", "Microwave", "Hair dryer", "Trash compactor", "Cooking basics", "Pool table", "Wine glasses", "Dishwasher", "Iron", "Outdoor dining area", "Keurig coffee machine", "Smoke alarm", "Keypad", "Barbecue utensils", "BBQ grill", "Outdoor furniture", "Board games", "Kitchen", "Ethernet connection", "Heating", "First aid kit", "Sound system", "Coffee maker", "Cable TV", "Toaster", "Air conditioning", "Extra pillows and blankets", "Backyard", "Essentials", "Pool", "Body soap", "Wifi", "Fire pit", "Luggage dropoff allowed", "Long term stays allowed", "Cleaning products", "Freezer", "Hangers", "Piano", "Indoor fireplace", "Lock on bedroom door", "Dedicated workspace", "Conditioner", "Mini fridge", "Carbon monoxide alarm", "Patio or balcony", "Free street parking", "Fire extinguisher", "Bathtub".

# In[151]:


ex4['amenities'].value_counts().idxmax()


# In[152]:


df = ex4['amenities'].value_counts()
df = df[:20]


# In[153]:


df


# In[32]:


da2 = ex4['amenities'].value_counts()
da2 = da2[:20]


# In[33]:


remove_words = ['allowed']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))

da2['amenities'] = ex4['amenities'].str.replace(pat, '', regex=True)


# In[34]:


da2['amenities'] = da2['amenities'].str.replace('parking','', regex=True).str.strip('parking')


# In[35]:


da2['amenities'] = da2['amenities'].str.replace('water','', regex=True).str.strip('water')


# In[36]:


da2['amenities'] = da2['amenities'].str.replace('Hot','hot-water')


# In[37]:


da2['amenities'] = da2['amenities'].str.replace('Free','free-parking')


# In[38]:


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


# ## Results: these amenities are the most favorite ones in those rentals that have best reviews. If someone would like to buys amenities for his rental, these ones can be as best profitable options to buy.

# ## Mission: Find amenities which are most favorite ones based on demanding:

# In[42]:


da3 = ex2['amenities'].value_counts()
da3 = da3[:20]


# In[43]:


remove_words = ['allowed']
pat = r'\b(?:{})\b'.format('|'.join(remove_words))

da3['amenities'] = ex2['amenities'].str.replace(pat, '')


# In[44]:


da3['amenities'] = da3['amenities'].str.replace('parking','', regex=True).str.strip('parking')


# In[45]:


da3['amenities'] = da3['amenities'].str.replace('Free','free-parking')


# In[165]:


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

# In[21]:


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

rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N)[3:8],
                    columns=['Word', 'Frequency']).set_index('Word')

matplotlib.style.use('ggplot')

#rslt.plot.bar(rot=0)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)
plt.title('Most Favorite Amenities')
####
df5 = AI['price'].mean()
print('The mean of rental price this neighborhood: ', round(df5))
print('The price description in this neighborhood:', AI['price'].describe())


# # Final Prt: AI for rental's price

# In[26]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


# In[30]:


val1 = input("Enter your neighborhood: ")
def prrice(xxx):
    data3 = data
    data3 = data3.dropna(axis=0, subset=['accommodates'])
    data3 = data3.dropna(axis=0, subset=['bedrooms'])
    data3 = data3.dropna(axis=0, subset=['beds'])
    data3 = data3.dropna(axis=0, subset=['numberofbath'])
    data3 = data3.dropna(axis=0, subset=['number_of_amenities'])
    data3 = data3.dropna(axis=0, subset=['price'])
    AI2 = data3[data3['neighbourhood_cleansed'].str.contains(xxx)]
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
    from sklearn.ensemble import RandomForestRegressor
    regressor2 = RandomForestRegressor(max_depth = 2, n_estimators = 20, random_state = 1)
    regressor2.fit(x_train, y_train.values.ravel())
    y_pred2ts = regressor2.predict(x_test)
    # Evaluating the Algorithm
    y_pred2tr = regressor2.predict(x_train)
    #print('R2 score of training set is {}',format((r2_score(y_train, y_pred2tr)/r2_score(y_train, y_pred2tr))*AI3))
    #print('R2 score of testing set is {}',format((r2_score(y_train, y_pred2ts)/r2_score(y_train, y_pred2ts))*AI7))
    val2 = int(input("Enter number of your accommodates: "))
    val3 = int(input("Enter number of your bedrooms: "))
    val4 = int(input("Enter number of your beds: "))
    val5 = int(input("Enter number of your bath: "))
    val6 = int(input("Enter number of your amenities: "))
    val_final = np.array([[val2, val3, val4, val5, val6]]) 
    #print(val_final)
    #y_trainP2 = poly_model.predict(val_final)
    
    y_pred2ts_j = regressor2.predict(val_final)
    round_off_values2 = np.around(y_pred2ts_j, decimals = 2)
    r2t = r2_score(y_train, y_pred2tr)
    r2te = r2_score(y_test, y_pred2ts)
    print('R2 score of training set is ',format(r2t))
    print('R2 score of testing set is ',format(r2te))
    print ("\nThe best price for your rental in USD : \n", round_off_values2)
    
prrice(val1)
    


# In[32]:


get_ipython().system('pip install xgboost==1.0.1')


# ## Just for test

# In[33]:


import xgboost


# In[34]:


from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


# In[37]:


##Test
val1 = input("Enter your neighborhood: ")
def pprice(xx):
    data22 = data
    data22 = data22.dropna(axis=0, subset=['accommodates'])
    data22 = data22.dropna(axis=0, subset=['bedrooms'])
    data22 = data22.dropna(axis=0, subset=['beds'])
    data22 = data22.dropna(axis=0, subset=['numberofbath'])
    data22 = data22.dropna(axis=0, subset=['number_of_amenities'])
    data22 = data22.dropna(axis=0, subset=['price'])
    AI22 = data22[data22['neighbourhood_cleansed'].str.contains(xx)]
    AI22 = AI22[(AI22['demanding_rate']>=100)]
    AI22 = AI22[(AI22['review_scores_rating']>=5)]
    x1 = AI22[['accommodates','bedrooms','beds','numberofbath','number_of_amenities']]
    y1 = AI22[['price']]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    x_train1, x_test1 = train_test_split(x1, test_size=0.2, random_state=25)
    y_train1, y_test1 = train_test_split(y1, test_size=0.2, random_state=25)
    #print(x)
    #print(x.info())
    #print(y)
    #print(y.info())
    #regressor2 = RandomForestRegressor(random_state = 1)
    #regressor2.fit(x, y)
    model_xb_33 = MultiOutputRegressor(xgb.XGBRegressor(gamma=0,                 
                                                       learning_rate=0.01,
                                                       max_depth=10,
                                                       n_estimators=150,
                                                       subsample=0.8,
                                                       num_parallel_tree=1,
                                                       colsample_bylevel=1,
                                                       scale_pos_weight=1,
                                                       random_state=34))
    model_xb_33.fit(x_train1, y_train1)
    y_pred_train_xb_33 = model_xb_33.predict(x_train1)
    y_pred_test_xb_33 = model_xb_33.predict(x_test1)
    #print('R2 score of training set is ',format((r2_score(y_train, y_pred_train_xb_3)/r2_score(y_train, y_pred_train_xb_3))*AI3))
    #print('R2 score of testing set is ',format((r2_score(y_train, y_pred_train_xb_3)/r2_score(y_train, y_pred_train_xb_3))*AI4))

    val2 = int(input("Enter number of your accommodates: "))
    val3 = int(input("Enter number of your bedrooms: "))
    val4 = int(input("Enter number of your beds: "))
    val5 = int(input("Enter number of your bath: "))
    val6 = int(input("Enter number of your amenities: "))
    
    val_final = np.array([[val2, val3, val4, val5, val6]]) 
    #print(val_final)
    #y_trainP2 = poly_model.predict(val_final)
    y_pred_test_xb_3_j11 = model_xb_33.predict(val_final)

    #g = map('{:.2f}%'.format,y_pred_test_xb_3_j)
    #print(y_trainP2)
    #print('The best price for your rental is: {}'.format(y_pred_test_xb_3_j))
    #print(x)
    #print(x.info())
    #print(y)
    #print(y.info())
    #regressor2 = RandomForestRegressor(random_state = 1)
    #regressor2.fit(x, y)
    round_off_values11 = np.around(y_pred_test_xb_3_j11, decimals = 2)
    r2tt = r2_score(y_train1, y_pred_train_xb_33)
    r2tee = r2_score(y_test1, y_pred_test_xb_33)
    print('R2 score of training set is ',format(r2tt))
    print('R2 score of testing set is ',format(r2tee))
    print ("\nThe best price for your rental : \n", round_off_values11)
    
pprice(val1)


# ## As you can see on the rsults, the R2 scores are not satisfactory for the ML model. We solved this problem by using a bigger dataset. For copy write matter we are not allowed to release the final dataset and its structure and also improved back-end code.
# 
