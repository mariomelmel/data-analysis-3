import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_decision_list(serial):
    list_to_append = list()
    for element_serial in serial:
        if element_serial == 1:
            state = 0
        else:
            state = 1
        list_to_append.append(state)

    return list_to_append


# Draw Categorical Plot
def draw_cat_plot(df):
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke',
    # 'alco', 'active', and 'overweight'.

    df_cat = df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight', 'cardio']]
    print(df_cat)
    # df_cat_long = pd.melt(df_cat, var_name='variables', value_name='values')
    df_cat_long = pd.melt(df_cat, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight',
                                                                  'smoke'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one
    # of the columns for the catplot to work correctly.

    # Draw the catplot with 'sns.catplot()'

    graph_cat = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat_long, kind='count')
    graph_cat.set(ylabel="Total")
    plt.show()

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map(df):
    # Clean the data
    df = df.drop(df[df['ap_lo'] >= df['ap_hi']].index | df[df['height'] <= df['height'].quantile(0.025)].index |
                 df[df['height'] >= df['height'].quantile(0.975)].index |
                 df[df['weight'] <= df['weight'].quantile(0.025)].index |
                 df[df['weight'] >= df['weight'].quantile(0.975)].index)
    print('First: ', df)

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr_matrix)

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr_matrix.__round__(1), mask=mask, annot=True)
    plt.show()

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig


def main():
    # Import data
    df = pd.read_csv('medical_examination.csv')
    # Add 'overweight' column

    weight_serial = df['weight']
    height_serial = df['height']

    list_bmi = list()
    i = 0
    for element in weight_serial:
        BMI = element / ((height_serial[i] / 100) ** 2)
        if BMI > 25:
            calculate_overweight = 1
        else:
            calculate_overweight = 0
        list_bmi.append(calculate_overweight)
        i += 1

    over_weight = pd.Series(list_bmi, name='overweight')
    df['overweight'] = over_weight

    # Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1,
    # make the value 0. If the value is more than 1, make the value 1.

    df['cholesterol'] = pd.Series(create_decision_list(df['cholesterol']))
    df['gluc'] = pd.Series(create_decision_list(df['gluc']))

    draw_cat_plot(df)
    draw_heat_map(df)


if __name__ == "__main__":
    main()
