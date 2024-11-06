import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

print(iris_df.head())
print(iris_df.describe())
print(iris_df.corr())

def plot_histogram():
    plt.hist(iris_df['petal length (cm)'], bins=15, color='skyblue', edgecolor='black')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Petal Length')
    plt.show()

plot_histogram()

def plot_scatter():
    sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
    plt.title('Sepal Length vs Petal Length by Species')
    plt.show()

plot_scatter()

def plot_heatmap():
    correlation = iris_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

plot_heatmap()
