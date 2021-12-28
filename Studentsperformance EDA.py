#%%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore') 

# %%
#Load Data
df = pd.read_csv('StudentsPerformance.csv')
# %%
#EDA of data
df.sample(10)
# %%
list(df.columns)
# %%
df.isnull().sum()
#%%
df.head()
#%%
df.dtypes
# %%
df.shape
#%%
df.nunique()
# %%
def bar_plot(variable):
    var = df[variable]
    varValue = var.value_counts()
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,varValue))
# %%
sns.set_style('darkgrid')
categorical_variables = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for v in categorical_variables:
    bar_plot(v)
# %%
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with histogram".format(variable))
    plt.show()
# %%
numerical_variables = ['math score', 'reading score', 'writing score']
for m in numerical_variables:
    plot_hist(m)
# %%
#Basic Data Analysis
#Gender
df[["gender","math score"]].groupby(["gender"], as_index = False).mean().sort_values(by="math score",ascending = False)
# %%
df[["gender","reading score"]].groupby(["gender"], as_index = False).mean().sort_values(by="reading score",ascending = False)
# %%
df_scores = df[['gender', 'math score', 'reading score', 'writing score']]
df_grp = df_scores.groupby(['gender']).mean()
# %%
df_grp
# %%
df_grp.plot(kind = 'bar')
# %%
df_scores1 = df[['race/ethnicity', 'math score', 'reading score', 'writing score']]
df_grp1 = df_scores1.groupby(['race/ethnicity']).mean()
# %%
df_grp1
# %%
df_grp1.plot(kind = 'bar')
# %%
df_scores2 = df[['parental level of education', 'math score', 'reading score', 'writing score']]
df_grp2 = df_scores2.groupby(['parental level of education']).mean()
# %%
df_grp2
# %%
df_grp2.plot(kind = 'bar')
# %%
df_scores3 = df[['lunch', 'math score', 'reading score', 'writing score']]
df_grp3 = df_scores3.groupby(['lunch']).mean()
# %%
df_grp3
# %%
df_grp3.plot(kind = 'bar')
# %%
df_scores4 = df[['test preparation course', 'math score', 'reading score', 'writing score']]
df_grp4 = df_scores4.groupby(['test preparation course']).mean()
#%%
df_grp4
# %%
df_grp4.plot(kind = 'bar')
# %%
#Pie Chart
labels = df['race/ethnicity'].value_counts().index
sizes = df['race/ethnicity'].value_counts().values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Students by Races',color = 'black',fontsize = 15)

# %%
labels = df['gender'].value_counts().index
sizes = df['gender'].value_counts().values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Students by Genders',color = 'black',fontsize = 15)
#%%
#Violin Plot
plt.figure(figsize=(20,15))
sns.set_style(style="darkgrid")
plt.subplot(2,3,1)
sns.violinplot(x = 'gender', y = 'math score', data = df, palette="gist_ncar_r")
plt.subplot(2,3,2)
sns.violinplot(x = 'gender', y = 'reading score', data = df, palette="gist_ncar_r")
plt.subplot(2,3,3)
sns.violinplot(x = 'gender', y = 'writing score', data = df, palette="gist_ncar_r")
# %%
#Boxen Plot
import seaborn as sns
sns.set_style(style="darkgrid")


plt.figure(figsize=(20,15))

plt.subplot(2,3,1)
sns.boxenplot(x=df['lunch'], y=df['math score'],
              color="b", 
              scale="linear", data=df)

plt.subplot(2,3,2)
sns.boxenplot(x=df['lunch'], y=df['reading score'],
              color="b", 
              scale="linear", data=df)

plt.subplot(2,3,3)
sns.boxenplot(x=df['lunch'], y=df['writing score'],
              color="b", 
              scale="linear", data=df)
# %%
plt.figure(figsize=(20,15))

plt.subplot(2,3,1)
sns.swarmplot(x="test preparation course", y="math score",hue="parental level of education", data=df, palette="PRGn")

plt.subplot(2,3,2)
sns.swarmplot(x="test preparation course", y="reading score",hue="parental level of education", data=df, palette="PRGn")

plt.subplot(2,3,3)
sns.swarmplot(x="test preparation course", y="writing score",hue="parental level of education", data=df, palette="PRGn")

plt.show()
# %%
#Bar plot showing the distribution in a given categorical variable
counts = df['parental level of education'].value_counts()

plt.figure(figsize=(10,7))
sns.barplot(x=counts.index, y=counts.values, palette="Set3")

plt.ylabel('Number of Degree')
plt.xlabel('Degrees', style = 'normal', size = 24)

plt.xticks(rotation = 45, size = 12)
plt.yticks(rotation = 45, size = 12)

plt.title('Distribution of Parental Level of Education',color = 'black',fontsize=15)
plt.show()
# %%
df['overall'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
 
# %%
#boxplot
sns.set_style(style="darkgrid")

plt.figure(figsize=(20,15))

plt.subplot(2,2,1)
sns.boxplot(x="gender", y="overall", hue="test preparation course", data=df)

plt.subplot(2,2,2)
sns.boxplot(x="gender", y="overall", hue="parental level of education", data=df)

plt.subplot(2,2,3)
sns.boxplot(x="gender", y="overall", hue="lunch", data=df)

plt.subplot(2,2,4)
sns.boxplot(x="gender", y="overall", hue="race/ethnicity", data=df)

sns.despine(offset=10, trim=True)
plt.show()
# %%
#Scatter Plot with Marginal Ticks
g = sns.JointGrid(data=df, x="overall", y="math score", space=0, ratio=17)
g.plot_joint(sns.scatterplot, color="g", alpha=.6, legend=False)
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)
# %%
g = sns.JointGrid(data=df, x="overall", y="reading score", space=0, ratio=17)
g.plot_joint(sns.scatterplot, color="r", alpha=.6, legend=False)
g.plot_marginals(sns.rugplot, height=1, color="r", alpha=.6)
# %%
g = sns.JointGrid(data=df, x="overall", y="writing score", space=0, ratio=17)
g.plot_joint(sns.scatterplot, color="b", alpha=.6, legend=False)
g.plot_marginals(sns.rugplot, height=1, color="b", alpha=.6)
# %%

# %%

# %%

# %%
