import pandas as pd

from plotter import plot_data_1, plot_data_2


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
        save = False
        plot_data_1(df_indo, save)
        plot_data_2(df_indo, 'new_cases', 'blue', save)
        plot_data_2(df_indo, 'new_deaths', 'red', save)
