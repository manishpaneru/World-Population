# This is the Explorative data analysis that we are gonna do in world population data, we are gonna use matpotlib pandas and seabron primarily t odo
# the EDA project , we will be cleaning the data first then we are gonna do the correlation # First, let's import the necessary libraries and dependencies.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from matplotlib import figure

import seaborn as sns  # now let's import the data that we need and we are gonna have a look

df = pd.read_csv("world_population.csv")
df.head()  # Let's have a look at all of the data
pd.set_option("display.max_rows", None)
df  # Let's see some basic description of the data shall we l
df.describe()  # Let's check for the missing data first
# I have used different  ways to do this in my data cleaning projects
# I am gonna create a loop that will skim through all the columns looking for the missing data
for col in df.columns:  # this will loop through the columns
    pct_missing = np.mean(
        df[col].isnull()
    )  # This will store the percentage of the missing values from column and save it in pct_missing
    print(
        "{} - {:.2f}%".format(col, pct_missing * 100)
    )  # Well this is just the print statement #First let's take care of the null vlaues, let's just use mean to fill the data as we will have more data to analyze, as the name of countries
# And continents are in string let's just write the names of the columns in a list and loop through the list , or we could just index them
columns = [
    df.columns[5:17]
]  # this holds every single columns in the dataframe, let's run the loop
for col in columns:
    df[col] = df[col].fillna(df[col].mean())
# Let's see if it worked with our previous loop
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print("{} - {:.2f}%".format(col, pct_missing * 100))
# Seems like it did work#Let's do some more formatting and data cleaning
# First let's look at the data types of each columns
print(
    df.dtypes
)  # Soo we can notice from df.head() above that the data has a decimal point nad 0  at the end of some population columns, and as we know there are only whole
# number population possible so we dont need .0 at the end, so let's get rid of that
# Because we have multiple columns that need to be changes let's use another loop
columns = [df.columns[5:17]]
# This loop must loop through every columns and change the data type in each column
for col in columns:
    df[col] = df[col].astype("int64")
# Let's see if it worked
df
# Okay it did #One more thing that you might notice is that we also have country code in CCA3 columns which we don't necassarily need soo maybe get rid of that too
# Let's just drop the columns
df = df.drop(columns=["CCA3"])
df  # also the name of the column that holds the country names is country/territory we maybe we can clean that up too.
df.rename(columns={"Country/Territory": "Country"}, inplace=True)
df
# Now that that's done let's move on. #Now let's order the data by largest population in 2022 as it is the most recent data that we have
df.sort_values(
    by=["2022 Population"], inplace=False, ascending=False
)  # Let's see how many columns actually have duplicates in this dataframe, as it is population data have duplicates are unlikely
df.duplicated().sum()
# Soo yeah 0 duplicates #But incase if we had duplicates I just want to show case show to drop duplicates in a cool way.
# Let's just use the loop again.
for col in df.columns:
    df[col] = df[col].drop_duplicates()
# This should take care of the duplicates in any columns #So we dont need the country continent and capital data to find corolation so let's create a new dataframe without them
coo = df.drop(df.columns[1:4], axis=1)
coo.corr()  # Now that we have the corolation between 2 columns. let's visuzlize the corolation that would be really helpful for understanding the corolation.
# Let's use an heatmap
sns.heatmap(coo.corr(), annot=True)

plt.rcParams["figure.figsize"] = (25, 20)
plt.show()
# This heat map makes it much more easier to under the corolation between different olumns where lighter it is , lower the corolation #Let's create a box chart  that can show the top 10 populaiton by country.
df_sorted = df.sort_values(by="2022 Population", ascending=False)
N = 10
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sorted.head(N), x="Country", y="2022 Population")

plt.title("Top {} Countries by Population in 2022".format(N))
plt.xlabel("Country", fontsize=14)
plt.ylabel("Population", fontsize=14)
plt.xticks(rotation=45)
plt.show()  # We can also see the world population percentage distribution in a pie chart
df_continent = df.groupby("Continent")["World Population Percentage"].mean()
plt.pie(df_continent, labels=df_continent.index, autopct="%1.1f%%", startangle=140)
plt.title("World Population Percentage by Continent")
plt.axis("equal")
plt.show()
# Let's create a scatter plot that will show distribution of population vs area by continent
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="2022 Population",
    y="Area (km²)",
    hue="Continent",
    size="Density (per km²)",
    sizes=(20, 200),
)
plt.title("Population vs. Area by Continent")
plt.xlabel("Population")
plt.ylabel("Area (km²)")
plt.legend(title="Continent")
plt.show()
