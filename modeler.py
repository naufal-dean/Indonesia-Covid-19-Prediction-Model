import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit


class Modeler:
    """Modeler class implementation."""

    def __init__(self, df_indo):
        self.df_indo = df_indo
        self.sigmoid_coeff = None

    def sigmoid(self, x, L, x0, k, c):
        return (L / (1 + np.exp(-k * (x - x0))) + c)

    def get_regression_model(self):
        x = self.df_indo['date'].transform(
            lambda d: datetime.toordinal(d)
        ).values
        y = self.df_indo.iloc[:, 3].values
        # Initial guess
        p0 = [max(y), np.median(x), 1, min(y)]
        # Use logistic regression (sigmoid function)
        self.sigmoid_coeff, _ = curve_fit(self.sigmoid, x, y, p0)
        return self.sigmoid_coeff

    def plot_observed_total_case(self, fig, ax):
        # Plot line data
        ax.bar(self.df_indo['date'].values, self.df_indo['total_cases'],
               color='blue')
        # Return
        return fig, ax

    def plot_expected_total_case(self, fig, ax, pred_d):
        x = self.df_indo['date'].transform(
            lambda d: datetime.toordinal(d)
        ).values
        # Next n day, to be predicted
        x_pred = np.append(
            x,
            [i for i in range(x[-1] + 1, x[-1] + pred_d + 1)]
        )
        # Predict using the sigmoid curve
        if self.sigmoid_coeff is None:
            self.get_regression_model()
        plt.plot(x_pred, self.sigmoid(x_pred, *self.sigmoid_coeff),
                 color='red', linewidth=4)
        ax.set_ylim(bottom=0)
        # Return
        return fig, ax

    def plot_observed_and_expected_total_case(self, pred_d, save):
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        # Plot data
        fig, ax = self.plot_observed_total_case(fig, ax)
        fig, ax = self.plot_expected_total_case(fig, ax, pred_d)
        # Setup axes format
        ax.set(
            xlabel='Tanggal',
            ylabel='Jumlah'
        )
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        # Setup figure
        fig.autofmt_xdate()
        fig.suptitle('Prediksi Total Kasus Positif Covid-19 di Indonesia\n'
                     f'{pred_d} Hari Sejak Tanggal 5 Mei 2020', fontsize=20)
        # Setup legend
        blue_patch = mpatches.Patch(color='blue', label='Data aktual')
        red_line = mlines.Line2D([], [], color='red', label='Prediksi')
        plt.legend(handles=[blue_patch, red_line])
        # Show
        if (save):
            fig.savefig('plt/plot_regression_total_case.jpg')
        else:
            plt.show()

    def plot_observed_new_case(self, fig, ax):
        # Plot line data
        ax.bar(self.df_indo['date'].values, self.df_indo['new_cases'],
               color='blue')
        # Return
        return fig, ax

    def plot_expected_new_case(self, fig, ax, pred_d):
        x = self.df_indo['date'].transform(
            lambda d: datetime.toordinal(d)
        ).values
        # Predict using the sigmoid curve
        if self.sigmoid_coeff is None:
            self.get_regression_model()
        # Next n day, to be predicted
        x_pred = np.append(
            x,
            [i for i in range(x[-1] + 1, x[-1] + pred_d + 1)]
        )
        # y value predicted
        total_case = self.sigmoid(x_pred, *self.sigmoid_coeff)
        new_case = [0]
        for i in range(1, len(total_case)):
            if total_case[i] < 0 or total_case[i - 1] < 0:
                new_case.append(0)
            else:
                new_case.append(total_case[i] - total_case[i - 1])
        new_case = np.array(new_case)
        plt.plot(x_pred, new_case, color='red', linewidth=4)
        ax.set_ylim(bottom=0)
        # Return
        return fig, ax

    def plot_observed_and_expected_new_case(self, pred_d, save):
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        # Plot data
        fig, ax = self.plot_observed_new_case(fig, ax)
        fig, ax = self.plot_expected_new_case(fig, ax, pred_d)
        # Setup axes format
        ax.set(
            xlabel='Tanggal',
            ylabel='Jumlah'
        )
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        # Setup figure
        fig.autofmt_xdate()
        fig.suptitle('Prediksi Kasus Positif Baru Covid-19 di Indonesia\n'
                     f'{pred_d} Hari Sejak Tanggal 5 Mei 2020', fontsize=20)
        # Setup legend
        blue_patch = mpatches.Patch(color='blue', label='Data aktual')
        red_line = mlines.Line2D([], [], color='red', label='Prediksi')
        plt.legend(handles=[blue_patch, red_line])
        # Show
        if (save):
            fig.savefig('plt/plot_regression_new_case.jpg')
        else:
            plt.show()


if __name__ == '__main__':
    print('\n' * 4)

    # Filter dataframe
    pd.options.display.max_columns = None
    df = pd.read_csv('data/owid-covid-data.csv')
    columns_filter = ['iso_code', 'location', 'date', 'total_cases',
                      'new_cases', 'total_deaths', 'new_deaths']
    df_indo = df[df['location'] == 'Indonesia'][columns_filter]
    df_indo['date'] = pd.to_datetime(df_indo['date'], format='%Y-%m-%d')
    df_indo = df_indo.iloc[65:]
    # print(df_indo)

    m = Modeler(df_indo)
    save = True
    # save = False
    m.plot_observed_and_expected_total_case(90, save)
    m.plot_observed_and_expected_new_case(90, save)