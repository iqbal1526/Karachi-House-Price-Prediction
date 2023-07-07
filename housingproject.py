#!/usr/bin/env python
# coding: utf-8

# In[194]:


import numpy as np


# In[195]:


import pandas as pd

df = pd.read_csv(r'data.csv')


# <h3>first rows of dataset</h3>

# In[196]:


df.head()


# In[197]:


df.shape


# In[198]:


df.describe()


# <h3>dropping the unnecessary columns<h3>

# In[199]:


df = df.drop(labels = ['title', 'date added','url'], axis = 1)
df.head()


# <h3>since we want to predict the price of houses and flats so we will filter our dataset in that way<h3>

# In[200]:


df = df[df['type'].isin(['House','Flat'])]
df.head()


# In[201]:


df.shape


# <h3> we notice that area has missing values in some rows and and bathroom and bedrooms has "0" value in some rows which is not possible for a house or flat to have zero bedrooms and bathrooms and for this purpose we will split our data containing zero values of the bathroom and bedroom and clean that dataset further.</h3>

# In[202]:


uncleaned_data = df[(df['bedrooms'] == 0) | (df['bathrooms'] == 0)]

# data containing all the non zero values of bedroom and bathroom
df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]


# In[203]:


df.head()


# In[204]:


uncleaned_data.head()


# In[205]:


print(df.shape)
print(uncleaned_data.shape)


# <h3> we notice that in our uncleaned data the keywords column contain the bedrooms and bathrooms value but the data is missing in the bedroom and bathroom column. so first we remove the data that contains null value of keyword and then proceed further</h3>

# In[206]:


#checking how many null values are present in keywords column
uncleaned_data['keywords'].isna().sum()


# In[207]:


#removing the null values
uncleaned_data = uncleaned_data.dropna(subset=['keywords'])
uncleaned_data['keywords'].isna().sum()


# <h3>notice that some of the entries in keywords are like this: (Bedrooms: 3,Bathrooms: 3,Kitchens: 2) so we will extract the values of bathrooms and bedroom and convert it in int form and then store it in bedrooms and bathrooms column.</h3>

# In[208]:


uncleaned_data.dtypes


# In[209]:


import re


# In[210]:


# Define a function to extract the integer value from a string
def extract_integer(text):
    try:
        value = int(re.search(r'\d+', text).group())
        return value
    except AttributeError:
        return None

# Extract bedrooms and bathrooms values from "keywords" column
uncleaned_data['bedrooms'] = uncleaned_data['keywords'].apply(lambda x: extract_integer(x.split(',')[0]) if 'Bedrooms: ' in x else None)
uncleaned_data['bathrooms'] = uncleaned_data['keywords'].apply(lambda x: extract_integer(x.split(',')[1]) if 'Bathrooms: ' in x else None)

# Print the modified DataFrame
uncleaned_data.head()


# In[211]:


print(uncleaned_data['bathrooms'].isna().sum())
print(uncleaned_data['bedrooms'].isna().sum())


# <h3>we notice that there are still missing values in bedroom and bathrooms column so we will drop these rows.</h3>

# In[212]:


uncleaned_data.dropna(subset=['bathrooms', 'bedrooms'], inplace=True)
print(uncleaned_data['bathrooms'].isna().sum())
print(uncleaned_data['bedrooms'].isna().sum())


# In[213]:


uncleaned_data.shape


# <h3>joining the now cleaned data again with our original df<h3>

# In[214]:


df = pd.concat([df, uncleaned_data], ignore_index=True)
print(df.shape)
df.head()


# <h3>we notice that there are few row with null value in their area. so we separate the rows with the value of area as null to further analyse it and clean it</h3>

# In[215]:


df['area'].isna().sum()


# In[216]:


# df with all null area value
df_with_null_area = df[df['area'].isnull()]

# df without any null area value
df = df[~df['area'].isnull()]


# In[217]:


print(df_with_null_area['area'].isna().sum())
print(df['area'].isna().sum())


# In[218]:


print(df.shape)
print(df_with_null_area.shape)


# In[219]:


df_with_null_area.head()


# <h3>we notice that some information about area is stored in "description" column around the word "SQ YARD" so we create a regular expression to extract the value and store it in "area" column</h3>

# In[220]:


#pattern = r'(\d+) sq yards|(\d+) SQ YARDS|(\d+) Sq Yards|(\d+) yards|(\d+) YARDS |(\d+)SQ YARDS'
pattern = r'(\d+)\s*(?:sq yards?|SQ YARDS?|Sq Yards?|yards?|YARDS?)\s*(?:,\s*(\d+))?'

# Extract the area values from the descriptions using the regular expression pattern
df_with_null_area['area'] = df_with_null_area['description'].str.extract(pattern, flags=re.IGNORECASE).replace(pd.NA, np.nan).astype(float).max(axis=1)

# Print the updated DataFrame
df_with_null_area.head()


# In[221]:


print(df_with_null_area['area'].isna().sum())
print(df_with_null_area['area'].unique())


# In[222]:


df_with_null_area.head()


# <h3>there are still missing values present in area and we notice that some information of area is still present in "description" column in sq feet value, so we extract that value through regular expression, convert it in square yards and store it in "area" column</h3>

# In[223]:


pattern_sqft = r'(\d+)\s*(?:SQ FEET?|Sq Feet?|sq feet?|sqfeet?|SQFEET?|feet)\s*(?:,\s*(\d+))?'

# Extract the area values in square feet from the descriptions using the regular expression pattern
def extract_and_convert_area(description):
    match = re.search(pattern_sqft, description, flags=re.IGNORECASE)
    if match:
        area_sqft = float(match.group(1))
        area_yards = area_sqft * 0.111111
        return area_yards
    else:
        return None

mask = df_with_null_area['area'].isnull()

# Apply the function to the 'description' column and assign the converted values to the 'area' column
df_with_null_area.loc[mask, 'area'] = df_with_null_area.loc[mask, 'description'].apply(extract_and_convert_area)


# In[224]:


df_with_null_area.head()


# In[225]:


print(df_with_null_area['area'].isna().sum())
print(df_with_null_area['area'].unique())


# <h3>we remove the remaining rows with area as null values</h3>

# In[226]:


df_with_null_area.dropna(subset=['area'], inplace=True)
print(df_with_null_area['area'].isna().sum())


# In[227]:


print(df_with_null_area.shape)
df_with_null_area.head()


# <h3>we notice that df_with_null_area has area datatype as float and df has area datatype as object</h3>
# 

# In[228]:


df_with_null_area.dtypes


# In[229]:


df.dtypes


# <h4>converting df area column into int</h4>

# In[230]:


df['area'] = df['area'].str.replace(' Sq. Yd.', '')


# In[231]:


df['area'] = df['area'].str.replace(',', '')

df['area'] = df['area'].astype(int)


# <h4>converting df_with_null_area 'area' column into int</h4>
# 

# In[232]:


df_with_null_area['area'] = df_with_null_area['area'].astype(int)


# <h3>merging the original dataframe and data_with_null_area dataframe</h3>

# In[233]:


df = pd.concat([df, df_with_null_area], ignore_index=True)
print(df.shape)
df.head()


# <h3>we will be using location too while creating the application but it has missing values so we will use the information provided in "complete location to fill the location column"</h3>

# In[234]:


df['location'].isna().sum()


# In[235]:


df.loc[df['complete location'].str.contains('dha', case=False, na=False), 'location'] = 'DHA'

# Replace the value of "location" with 'bahria town' where "complete location" contains 'bahria town'
df.loc[df['complete location'].str.contains('bahria town', case=False, na=False), 'location'] = 'Bahria Town'

# Replace the value of "location" with 'clifton' where "complete location" contains 'clifton'
df.loc[df['complete location'].str.contains('clifton', case=False, na=False), 'location'] = 'Clifton'

df.head()


# In[236]:


print(df['location'].isna().sum())
print(df.shape)


# <h3>since we dont need the 'keywords', 'description' and 'complete location' columns so we will remove them</h3>

# In[237]:


df = df.drop(['complete location','description','keywords'],axis=1)
df.head()


# In[238]:


df.shape


# In[239]:


df.dtypes


# <h3>converting bedrooms, bathrooms and price datatype as int and float</h3>

# In[240]:


df[['bedrooms','bathrooms']] = df[['bedrooms','bathrooms']].astype(int)


# In[241]:


df.dtypes


# In[242]:


df['price'] = df['price'].str.replace(' Crore', '')


# <h4>converting values stored as lakh in "price" to "crore"</h3>

# In[243]:


df.loc[df['price'].astype(str).str.contains('lakh', case=False), 'price'] = df.loc[df['price'].astype(str).str.contains('lakh', case=False), 'price'].str.replace(r'\D+', '', regex=True).astype(float) / 100


# In[244]:


df.head()


# In[245]:


df['price'] = df['price'].astype(float)
print(df.dtypes)
df.head()


# <h3>converting categorical value to numerical value</h3>

# In[246]:


df_type = pd.get_dummies(df['type'])

df = pd.concat([df, df_type], axis=1)
df.head()


# In[247]:


df['price'].max()


# In[248]:


df_location = pd.get_dummies(df['location'])

df = pd.concat([df, df_location], axis=1)
df.head()


# In[249]:


print(df.shape)
print(df.dtypes)


# <h3>removing type and location column from the dataset as we dont need them anymore</h3>

# In[250]:


df = df.drop(['type','location'],axis=1)


# In[251]:


df.head()


# <h3>renaming columns to appropriate column names</h3>

# In[252]:


column_mapping = {
    'price': 'price(crore)',
    'area': 'area(sq. yard)'
}
df.rename(columns=column_mapping, inplace=True)


# <h3>A final look on our clean dataset</h3>

# In[253]:


print('rows and columns: ', df.shape,'\n')
print(df.dtypes)
df.head()


# In[254]:


df['price(crore)'].max()


# In[255]:


df.describe()


# In[ ]:





# <h3>removing outliers from price and area columns</h3>

# In[256]:


Q1a = df['price(crore)'].quantile(0.25)
Q3a = df['price(crore)'].quantile(0.75)
IQR1 = Q3a - Q1a

# Define the lower and upper bounds for outliers of price
lower_bound1 = Q1a - 1.5 * IQR1
upper_bound1 = Q3a + 1.5 * IQR1


Q1b = df['area(sq. yard)'].quantile(0.25)
Q3b = df['area(sq. yard)'].quantile(0.75)
IQR2 = Q3b - Q1b

# Define the lower and upper bounds for outliers for area
lower_bound2 = Q1b - 1.5 * IQR2
upper_bound2 = Q3b + 1.5 * IQR2


#removing outliers of area and price
df = df[((df['area(sq. yard)'] >= lower_bound2) & (df['area(sq. yard)'] <= upper_bound2)) & ((df['price(crore)'] >= lower_bound1) & (df['price(crore)'] <= upper_bound1))]
df.shape


# In[187]:


df['price(crore)'].max()


# <h3>plotting pearson correlation btw area and price to indicate relationship strength between these attributes</h3>

# In[188]:


import matplotlib.pyplot as plt
import seaborn as sns

x = df['price(crore)']
y = df['area(sq. yard)']

# Calculate the Pearson correlation coefficient
corr_coeff = df['price(crore)'].corr(df['area(sq. yard)'])

# Create a scatter plot
plt.scatter(x, y, alpha=0.5)

# Add the Pearson correlation line
sns.regplot(x=x, y=y, scatter=False, color='red')

# Add the correlation coefficient as a text annotation
plt.text(x.min(), y.max(), f'Correlation: {corr_coeff:.2f}', ha='left', va='top')

# Set labels and title
plt.xlabel('Price')
plt.ylabel('Area')
plt.title('Scatter Plot with Pearson Correlation Line')

# Show the plot
plt.show()


# <h3>applying regression tree and calculating value of MSE RMSE R2 of the model</h3>

# In[189]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['bedrooms', 'bathrooms', 'area(sq. yard)', 'House', 'Flat','Bahria Town','Clifton','DHA']], df['price(crore)'], test_size=0.2, random_state=42)

# Initialize and train the decision tree regression model
modelDTR = DecisionTreeRegressor()
modelDTR.fit(X_train, y_train)

# Make predictions on the test set
predictions = modelDTR.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r_squared = modelDTR.score(X_test, y_test)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r_squared)


# In[190]:


# Prepare the new input data
new_data = [[5, 5, 600, 1, 0, 0, 0, 1]]  # Example input data with feature values

# Make predictions on the new input data
predicted_price = modelDTR.predict(new_data)

# Print the predicted price
print("Predicted Price:", predicted_price)


# In[191]:


# Plotting the actual prices
plt.scatter(y_test, y_test, color='blue', label='Actual')

# Plotting the predicted prices
plt.scatter(y_test, predictions, color='red', label='Predicted')

# Add labels and title
plt.xlabel('Actual Prices (crore)')
plt.ylabel('Predicted Prices (crore)')
plt.title('Decision Tree Regression')

# Add legend
plt.legend()

# Display the chart
plt.show()


# <h2>applying random forest regressor and calculating values of MSE RMSE and R-Squared.</h2>

# In[192]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(df[['bedrooms', 'bathrooms', 'area(sq. yard)', 'Flat', 'House','Bahria Town','Clifton','DHA']], df['price(crore)'], test_size=0.2, random_state=42)
modelRFG = RandomForestRegressor()
modelRFG.fit(X_train, y_train)
predictions = modelRFG.predict(X_test)


# In[193]:


mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r_squared = modelRFG.score(X_test, y_test)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r_squared)


# In[ ]:


# Prepare the new input data
new_data = [[3, 4, 500, 1, 0, 1, 0, 0]]  # Example input data with feature values

# Make predictions on the new input data
predicted_price = modelRFG.predict(new_data)

# Print the predicted price
print("Predicted Price:", predicted_price)


# In[ ]:


# Plotting the actual prices
plt.scatter(y_test, y_test, color='blue', label='Actual')

# Plotting the predicted prices
plt.scatter(y_test, predictions, color='red', label='Predicted')

# Add labels and title
plt.xlabel('Actual Prices (crore)')
plt.ylabel('Predicted Prices (crore)')
plt.title('Forest Tree Regression')

# Add legend
plt.legend()

# Display the chart
plt.show()


# <h2>we notice that the random forest regressor yields the best result with minimum Mean Squared Error value and maximum R Squared value, so we will choose this model for our prediction</h2>

# <h3>For Frontend purposes</h3>

# In[ ]:


import pickle

# Assuming your trained model is stored in the variable `model`
with open('housingProject.pickle', 'wb') as f:
    pickle.dump(modelRFG, f)


# In[ ]:


import json
columns = {
    'data_columns': [col.lower() for col in df.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))

