# This is a Explorative data analysis portfolio project that i am gonna include in my portfolio. This uses a world population dataset from
# upuntil 2023. I am gonna use this dataset to showcase my data cleaning and explorative analysis skills


# First let's import all the libraries and dependencies that we need for this explorative data analysis project
# we need all the following libraries and dependencies for data cleaning as well as visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import altair as alt
from scipy.stats.mstats import winsorize

df = pd.read_csv("world_population.csv")


# First let's see some basic info of the dataset we have here
df.info


# Now let's see some more indepth description of this dataset so we will have more idea of what we can do.
df.describe


# Now let's see the data types of each columns in the dataset
df.dtypes


# We can also see the different columns from dataframe using this code
print(df.columns)


# Now we are gonna remove rank and CCA3 columns as they are absolutely useless for our analysis
df.drop(["Rank", "CCA3"], axis=1, inplace=True)  # Assuming these aren't needed


# Now let's rename Density(per km2) to just density, as this will be easier to type and easier to understand too.
df.rename(columns={"Density (per km²)": "density"}, inplace=True)


# Now let's see what percentage of null value in each columns from this dataset
for column in df.columns:
    null_percentage = df[column].isnull().mean() * 100
    print(f"Column {column}: {null_percentage:.2f}% null values")


# let's fill the null values of all the columns, using backfill as the country with the nearest data just below the null value column won't effect us much.
df = df.fillna(method="bfill")


#  Let's rename country/Territory columns to Just country as this will be much easier, similarly Area(km2) will be renamed to Area
# As this will be much easier for our future analysis
df.rename(columns={"Country/Territory": "Country", "Area (km²)": "Area"}, inplace=True)


#  let's convert population columns to integers so that it would be easier to do data cleanign and calculation for this data frame.
for col in [
    "2022 Population",
    "2020 Population",
    "2015 Population",
    "1990 Population",
    "1970 Population",
]:
    df[col] = df[col].astype("int64")


# Outlier Management: Cap '2022 Population' at 99th Percentile
# Extremely large populations (outliers) can skew our analysis. To mitigate this, I'll cap
# the '2022 Population' at the 99th percentile, retaining the vast majority of data points
# while reducing the impact of extreme values.
percentile_99 = df["2022 Population"].quantile(0.99)
df["2022 Population"] = df["2022 Population"].clip(upper=percentile_99)
print(percentile_99)


# Now we need to create a column called standardized growth rate That will have standrad diveation range for each population growth rate for the data
# This code facilitates comparative analysis of population growth across countries with varying scales, I am standardizing the 'Growth Rate' data. This transformation
# will center the distribution around zero and scale it to have unit variance, ensuring that the magnitude of growth rates is relative and comparable.
scaler = StandardScaler()
df["Standardized_Growth_Rate"] = scaler.fit_transform(df[["Growth Rate"]])
print(df["Standardized_Growth_Rate"])


# Now we are gonna create different groups of continents as this will make it much more easier for our further analysis.
df["Region"] = df["Continent"].map(
    {
        "Asia": "Asia",
        "Africa": "Africa",
        "Europe": "Europe",
        "North America": "Americas",
        "South America": "Americas",
        "Oceania": "Oceania",
        "Antarctica": "Other",
    }
)


# Now we are gonna delete rows of nations where the population  is less than 1 million, creating a threshold,  and then deleting rows with population less than the threshold
threshold = 1000000  # 1 million
df = df[df["2022 Population"] >= threshold]


#  Now we are gonna create an additional row that will hold the data population density data as it will help us with better analysis
df["Population_Density"] = df["2022 Population"] / df["Area"]


# Now this code will convert perform datetime formating in the rows in the data
for year in [1970, 1980, 1990, 2000, 2010, 2015, 2020]:
    df[f"{year}"] = pd.to_datetime(df[f"{year} Population"].astype(str), format="%Y")


# Now let's delete all the rows in the dataset with blank value in capital columns.
df.dropna(subset=["Capital"], inplace=True)


# Let's create a new column that holds the data of  population bins that will help us with further analysis
df["Population_Category"] = pd.cut(
    df["2022 Population"],
    bins=[0, 10000000, 50000000, 100000000, np.inf],
    labels=["Small", "Medium", "Large", "Very Large"],
)
print(df["Population_Category"])


# Let's delete duplicate rows from the dataset.
df.drop_duplicates(inplace=True)


# Now that the data cleaning and modification is done, let's reset the index so we can move on with the analysis
df.reset_index(drop=True, inplace=True)


# Now let's see the overview of the dataset , before we move on to visualization and analysis
df.head()


# Now let's create a scatter plot of Area vs. density so we better understand the correlation between the area of a country and population density
# Create a scatter plot of Area vs density
plt.figure(figsize=(8, 6))
plt.scatter(df["Area"], df["density"], color="blue", marker="o")

# Let's add relevant title and labels
plt.title("Scatter Plot of Area vs Density")
plt.xlabel("Area")
plt.ylabel("Density")


# This Scatter plot shows that there is minimal corroletion betweeen Area of a country and It's population density, As small nations can have huge
# population but also some country might be absolutely huge and have a comperetively small population


# Now let's create a scatter plot of Area vs. growth Rate of the population, THis will give us a better idea of whether the area has any influence over
# the Growth Rate of the population so we will know if a country being bigger or smaller in size will have any impact on population
# Create a scatter plot of Area vs GrowthRate
plt.figure(figsize=(8, 6))
plt.scatter(df["Area"], df["Growth Rate"], color="green", marker="x")

# Add title and labels
plt.title("Scatter Plot of Area vs GrowthRate")
plt.xlabel("Area")
plt.ylabel("Growth Rate")

# Show the plot
plt.grid(True)
plt.show()


# After seeing this scatter plot we can be more certain that population of a nation isn't mcuh related to area of the nation.
# But this scatter does make it clear that smalelr countrues tend to have higher population growth rate, This might also be because
# Most of this nation are third-world nation who are experiencing exponential Population Growth year over Year


# Let's create a scatter plot of density vs. Growth Rate so that we will know if a nation is experiencing high population growth , do they also
# experience rise in population density
plt.figure(figsize=(8, 6))
plt.scatter(df["density"], df["Growth Rate"], color="red", marker="o")

# Add title and labels
plt.title("Scatter Plot of Density vs Growth Rate")
plt.xlabel("Density")
plt.ylabel("GrowthRate")

# Show the plot
plt.grid(True)
plt.show()


# After seeing this scatter plot I can clearly identify that , Having a big population growth rate doesn't mean that populaiton density is high
# but can see that coutnries with higher population growthrate does seem to have low population density expect of some ouliers all the datapoints
# signal towards the same thing.


# Let's create a scatterplot of Population of 2022 Vs Density , so that we know higher population results in higher density or is there some other trend.
plt.figure(figsize=(8, 6))
plt.scatter(df["2022 Population"], df["density"], color="purple", marker="o")

# Add title and labels
plt.title("Scatter Plot of 2022 Population vs Density")
plt.xlabel("2022 Population")
plt.ylabel("Density")

# Show the plot
plt.grid(True)
plt.show()


# From Above scatter plot we can analyze that countries with small population usualy seems to have lowest density with some outliers
# although there are some countries with high population taht has low population density and also some countries with low population with high population density
# THat means it doesn't really have coroletion between population and desity of population


# now let's create a scatter plot with 2022 population vs Growth Rate so that we know if high population means higher growth rate.
plt.figure(figsize=(8, 6))
plt.scatter(df["2022 Population"], df["Growth Rate"], color="orange", marker="o")

# Add title and labels
plt.title("Scatter Plot of 2022 Population vs Growth Rate")
plt.xlabel("2022 Population")
plt.ylabel("GrowthRate")

# Show the plot
plt.grid(True)
plt.show()


# This scatter chart clearly indicates that there is no corolation between higher population and growth rate
# The average growth rate is almost similiar between all countries, although there are some countries with low population with high growth rate
# Also there are many countries with high population they have high population growth rate


# Now we are going to create a histogram of density over time better to understand density growth rate over time in the data.
plt.figure(figsize=(8, 6))
plt.hist(df["density"], bins=5, color="lightgreen", edgecolor="black")

# Add title and labels
plt.title("Histogram of Density")
plt.xlabel("Density")
plt.ylabel("Frequency")

# Show the plot
plt.grid(True)
plt.show()


# This histogram easily shows that the growth rate was high during the fisrt date data point in the dataset and then the growth rate steadly slows down.
# This could mean much more awarness about the increase of ppulation or people just being too poor to have children, especially in the first world nation.


# Create histogram of 2022 Population
hist, bin_edges = np.histogram(df["2022 Population"], bins=5)

# Plot line chart of histogram
plt.figure(figsize=(8, 6))
plt.plot(bin_edges[:-1], hist, marker="o", color="navy", linestyle="-")

# Add title and labels
plt.title("Line Chart of Histogram of 2022 Population")
plt.xlabel("2022 Population")
plt.ylabel("Frequency")

# Show the plot
plt.grid(True)
plt.show()


# Now we are gonna create a scatterplot that will show us the Growth rate of population and group them by countinents
# Define colors for each continent
colors = {
    "Asia": "red",
    "Europe": "blue",
    "Africa": "green",
    "Australia": "Pink",
    "South America": "Yellow",
    "North America": "Grey",
}

# Create scatter plot of World Population Growth vs Growth Rate for each continent
plt.figure(figsize=(8, 6))
for continent, color in colors.items():
    continent_data = df[df["Continent"] == continent]
    plt.scatter(
        continent_data["World Population Percentage"],
        continent_data["Growth Rate"],
        color=color,
        marker="o",
        label=continent,
    )

# Add legend
plt.legend(title="Continent")

# Add title and labels
plt.title("Scatter Plot of World Population Growth vs Growth Rate")
plt.xlabel("World Population Percentage")
plt.ylabel("Growth Rate")

# Show the plot
plt.grid(True)
plt.show()


# We can easily understand that Africa has the lowest world population percentage but also oe of the highest growth rate.
# but one thing bothering me, there are 2 countries in ASIA with most population, probably india and China, have huge population but below average Growth rate
# but on the contrary There is a european country with very small population but high Population growth rate, Similiarly
# There is a nation in europe which has small populaiton and very small Growth rate, Maybe it merits more research.


# We are gonna create a new bar diagram that holds the data of number of countries per continent and arranged from most to least
# Aggregate the data to get the count of countries per continent
country_count_by_continent = df[
    "Continent"
].value_counts()  # Removed ['Country/Territory']

# Define colors for each continent, including a default color
colors = {
    "Asia": "red",
    "Africa": "blue",
    "South America": "green",
    "North America": "purple",
    "Oceania": "orange",
    "Europe": "cyan",
    "default": "gray",
}

# Create bar chart of Number of Countries per Continent
plt.figure(figsize=(10, 6))
plt.bar(
    country_count_by_continent.index,
    country_count_by_continent.values,
    color=[
        colors.get(continent, colors["default"])
        for continent in country_count_by_continent.index
    ],
)

# Add title and labels
plt.title("Number of Countries per Continent")
plt.xlabel("Continent")
plt.ylabel("Number of Countries")

# Show the plot
plt.grid(axis="y")
plt.xticks(rotation=45)
plt.show()


# Seems like Asia, Even though is the largest continents have 2 most number of countries , Myabe due to presence of extemly large nations like india, Russia , China ETC
# Maybe due to this reason Africa has the highest number of countries , It maybe due to having smaller nations in Africa or , Maybe Colozinsation did it's magic.


# let's create a histogram that will show us the growth rate between the time period of our dataset, This will give us an idea of what population growth rate over time looks liek
plt.figure(figsize=(10, 6))
plt.hist(df["Growth Rate"], bins=20, alpha=0.6, color="skyblue")
plt.title("Distribution of Growth Rate")
plt.xlabel("Growth Rate")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.show()


# From the produced histogram we can easily identify that the histogram grew exponentially between 2018 to 2020 then
# it can be because of growth of population or much more consice and precise census during Covid during vaccination


# Now let's Create a scatter plot of 2022 Population VS Density
plt.figure(figsize=(10, 6))
plt.scatter(df["density"], df["2022 Population"], alpha=0.6)
plt.xscale("log")
plt.yscale("log")
plt.title("2022 Population vs. Density (per km²)")
plt.xlabel("Density (per km²)")
plt.ylabel("2022 Population")
plt.grid(True)
plt.show()


# This scatter plot can easily understand that The density of a countries population is in no way coroleted with a countries population
# As there are countries with high population with small desnity and vice versa. As this is evident Through the scatter plot as well as explorative data analysis


# this has been my first explorative data analysis project in python. I will upload the code in github and kaggle and the findings into portfolio website
# In this project we analyzed World population data we found on Kaggle. Link will be with the file in GitHub.
# I will see you in my next explorative data analysis project in SQL , Python and in tableau along with other ML skills.
# Please visit my portfolio website for more information about my portfolio project website.
