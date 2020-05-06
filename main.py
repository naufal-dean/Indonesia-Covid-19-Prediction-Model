import pandas as pd

from plotter import plot_data_1, plot_data_2
from modeler import Modeler


if __name__ == '__main__':
    plot = True
    # plot = False

    # Filter dataframe
    pd.options.display.max_columns = None
    df = pd.read_csv('data/owid-covid-data.csv')
    columns_filter = ['iso_code', 'location', 'date', 'total_cases',
                      'new_cases', 'total_deaths', 'new_deaths']
    df_indo = df[df['location'] == 'Indonesia'][columns_filter]
    df_indo['date'] = pd.to_datetime(df_indo['date'], format='%Y-%m-%d')
    # print(df_indo)

    # Plot data
    if plot:
        save = True
        # save = False
        # Plot actual data
        plot_data_1(df_indo.iloc[30:], save)
        plot_data_2(df_indo.iloc[30:], 'new_cases', 'blue', save)
        plot_data_2(df_indo.iloc[30:], 'new_deaths', 'red', save)
        # Plot prediction data
        m = Modeler(df_indo.iloc[65:])
        m.plot_observed_and_expected_total_case(90, save)
        m.plot_observed_and_expected_new_case(90, save)
