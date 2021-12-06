# Name: Sudar Kartheepan
# Date: December 6, 2021
# Data Science Project
# What do the 200 highest-grossing films have in common?
# I recommend viewing all the visualizations in full screen to understand them fully

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections

df = pd.read_csv("movies.csv")

# Deleting unecessary column from the dataframe
del df['votes']

# Sorting values according to gross and taking the first 200 values as the data
df = df.sort_values(["gross"], ascending=False, ignore_index=True)
data = df.head(200)

#data starts from index 1 now
data.index += 1

#company column
companies = data['company']

#finding the frequency of each production company in the column; taking the highest 12
company_frequency = companies.value_counts()
top_companies = company_frequency.head(12)

#extracting the exact number of times each of the 12 highest companies appear
comp_count = []
for value in top_companies:
    comp_count.append(value)

f = plt.figure()
#bar plot of the 12 companies; alternating colors
top_companies.plot.barh(width=0.5, fontsize=7, color=['#01afd1', '#f8e71c', '#01afd1', '#f8e71c', '#01afd1', '#f8e71c',
                                                       '#01afd1', '#f8e71c', '#01afd1', '#f8e71c', '#01afd1',
                                                       '#f8e71c'])

#plotting the frequency of each company on the graph
for i in range(len(top_companies)):
    plt.annotate(comp_count[i], (comp_count[i] + 0.1, -0.1 + i))

#title
plt.title("Production Companies of Highest Grossing Films")

#taking the genre column
genres = data['genre']

#top 3 genres
top_genres = genres.value_counts().head(3)

#adding "Other" as a genre
other_genres = pd.Series([200 - top_genres.sum()], index=["Other"])
main_genres = top_genres.append(other_genres)

#values for how many movies with each genre
genre_count = []
for value in main_genres:
    genre_count.append(value)

#function to display the value and percentage of each pie slice inside the slice itself
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct

#plotting pie chart
f1 = plt.figure()
plt.title("Genres of Highest Grossing Films")
main_genres.plot.pie(label="Genres", autopct=make_autopct(genre_count))

#getting movie titles
names = data["name"]

f2 = plt.figure()

#making a string of all the titles
text = " ".join(title for title in names)

#creating a wordcloud of the titles to show frequency of words and phrases appearing in titles
wordcloud = WordCloud(background_color="#f8e71c").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

f3 = plt.figure()
#get countries of release
top_countries = data['country'].value_counts()

#getting values of how many movies released in each country
country_count = []
for value in top_countries:
    country_count.append(value)

#bar plot of countries
top_countries.plot.bar(color=['#01afd1', '#f8e71c', '#01afd1', '#f8e71c', '#01afd1'])

#plotting number of movies per country on graph
for i in range(len(top_countries)):
    plt.annotate(country_count[i], (i - 0.025, country_count[i] + 1))
#rotate labels
plt.xticks(rotation=30)
plt.title("Release Countries of Highest Grossing Films")

#getting dates of movies
dates = data['released']

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']

#make a list of release months for all movies
month_list = []
for date in dates:
    month_list.append(date.split(' ')[0])

# used these below lines for getting values for number of movies released each month
# month_frequency = collections.Counter(frequency)
# print(month_frequency)

#list of the values
month_values = [2, 6, 13, 7, 38, 31, 34, 5, 1, 5, 34, 24]

f4 = plt.figure()
#regular line plot
plt.plot(months, month_values)
plt.title("Month of Release of Highest Grossing Films")

#getting the duration movies
duration = data['runtime']
f5 = plt.figure()

#boxplot of duration
bp = plt.boxplot(duration, vert=False, showmeans=True, patch_artist=True)

#setting the color of the boxplot
for box in bp['boxes']:
    box.set(color='#01afd1', linewidth=2)
    box.set(facecolor='#f8e71c')

#plotting values for maximum and minimum
for line in bp['whiskers']:
    #getting the x and y coordinates
    x, y = line.get_xydata()[1]
    plt.text(x, y+0.05, '%.1f' % x, horizontalalignment='center')

#plotting the values for any outliers
for point in bp['fliers']:
    x, y = point.get_xydata()[0]
    plt.text(x, y+0.02, '%.1f' % x, horizontalalignment='center')

#plotting the value of the median
for line in bp['medians']:
    x, y = line.get_xydata()[1]
    plt.text(x, y+0.01, '%.1f' % x, horizontalalignment='center')  # draw above, centered

#plotting the values for the 25th and 75th percentile
for box in bp['boxes']:
    x, y = box.get_path().vertices[0]  # bottom of left line
    plt.text(x, y-0.01, '%.1f' % x, horizontalalignment='center',
             verticalalignment='top')
    x, y = box.get_path().vertices[3]  # bottom of right line
    plt.text(x, y-0.01, '%.1f' % x,
             horizontalalignment='center',
             verticalalignment='top')

plt.title("Duration of Release of Highest Grossing Films")

plt.show()