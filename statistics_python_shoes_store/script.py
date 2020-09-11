import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv("shoes_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['SalePrice'] = df['SalePrice'].apply(lambda x: float(x[2:]))
df['UnitPrice'] = df['UnitPrice'].apply(lambda x: float(x[2:]))


describe = df.describe()


categorical_variables = ['Country', 'ProductID', 'Shop', 'Gender', 'Size (US)', 'Discount', 'Year', 'Month']
numerical_variables = ['UnitPrice', 'SalePrice']

frequency_shops = df['Shop'].value_counts().head(10)
df_frequency_shops = pd.DataFrame({'shops': frequency_shops.index.tolist(), 'quantity': frequency_shops.tolist()})
sns.barplot(x='shops', y='quantity', data=df_frequency_shops)
plt.show()

for cat_variable in categorical_variables:
    frequency = df[cat_variable].value_counts()
    df_frequency = pd.DataFrame({cat_variable: frequency.index.tolist(), 'quantity': frequency.tolist()})
    sns.barplot(x=cat_variable, y='quantity', data=df_frequency)
    plt.show()

for num_variable in numerical_variables:
    sns.histplot(df[num_variable], bins='auto')
    plt.show()


corr = df.corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True)
plt.show()

grouped = df[(df['Year'] != 2014) & (df['Gender'] == 'Male') & (df['Country'] == 'United States')]\
    .groupby(['Size (US)', 'Year', 'Month']).size().unstack(level=0).fillna(value=0)

means = []
standard_errors = []
for column in grouped.columns:
    means.append(grouped[column].mean())
    standard_errors.append(grouped[column].sem())


d = {'means': means, 'std_error': standard_errors}
df_calculations = pd.DataFrame(data=d, index=grouped.columns)

df_calculations['error_margin'] = df_calculations['std_error'].apply(lambda x: x * 2.07)
df_calculations['low_margin'] = df_calculations.apply(lambda x: x['means'] - x['error_margin'], axis=1)
df_calculations['up_margin'] = df_calculations.apply(lambda x: x['means'] + x['error_margin'], axis=1)
df_calculations['math_round_up'] = df_calculations.apply(lambda x: math.ceil(x['up_margin']), axis=1)

