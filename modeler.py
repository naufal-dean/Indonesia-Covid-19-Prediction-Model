import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


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

    def get_regression_total_case_accuracy(self):
        if self.sigmoid_coeff is None:
            self.get_regression_model()
        # Predict data
        x_data = self.df_indo['date'].transform(
            lambda d: datetime.toordinal(d)
        ).values
        y_data = self.df_indo['total_cases'].values
        y_pred = self.sigmoid(x_data, *self.sigmoid_coeff)
        print('RMSE (root mean squared error)',
              mean_squared_error(y_data, y_pred))
        print('R^2 (coefficient of determination)',
              r2_score(y_data, y_pred))

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
        # Predict using the sigmoid curve
        if self.sigmoid_coeff is None:
            self.get_regression_model()
        # Next n day, to be predicted
        x_pred = np.append(
            x,
            [i for i in range(x[-1] + 1, x[-1] + pred_d + 1)]
        )
        total_case = self.sigmoid(x_pred, *self.sigmoid_coeff)
        # Plot data
        plt.plot(x_pred, total_case, color='red', linewidth=4)
        ax.set_ylim(bottom=0)
        # Additional info
        print('Total case:')
        print(f'Total kasus positif {pred_d} hari dari tanggal 5 Mei 2020 =',
              int(total_case[-1]), 'orang')
        y_data = self.df_indo['total_cases'].values
        print('RMSE (root mean squared error)',
              mean_squared_error(y_data, total_case[:len(y_data)]))
        print('R^2 (coefficient of determination)',
              r2_score(y_data, total_case[:len(y_data)]))
        # Return
        return fig, ax

    def plot_observed_and_expected_total_case(self, pred_d, save):
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
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
        max_new_case = -1
        max_new_case_idx = -1
        for i in range(1, len(total_case)):
            if total_case[i] < 0 or total_case[i - 1] < 0:
                new_case.append(0)
            else:
                new_case_daily = total_case[i] - total_case[i - 1]
                new_case.append(new_case_daily)
                # Get max new case
                if new_case_daily > max_new_case:
                    max_new_case = new_case_daily
                    max_new_case_idx = i

        new_case = np.array(new_case)
        plt.plot(x_pred, new_case, color='red', linewidth=4)
        ax.set_ylim(bottom=0)
        # Aditional info
        print("New case:")
        print("Max new case in a day  =", int(max_new_case))
        print("Max new case date      =",
              str(datetime.fromordinal(x_pred[max_new_case_idx]))
              .split()[0])
        under_1_new_case = False
        under_5_new_case = False
        under_10_new_case = False
        for case, date in zip(reversed(new_case), reversed(x_pred)):
            if case > 1 and not under_1_new_case:
                under_1_new_case = True
                print("New case under 1 date  =",
                      str(datetime.fromordinal(date + 1))
                      .split()[0])
            if case > 5 and not under_5_new_case:
                under_5_new_case = True
                print("New case under 5 date  =",
                      str(datetime.fromordinal(date + 1))
                      .split()[0])
            if case > 10 and not under_10_new_case:
                under_10_new_case = True
                print("New case under 10 date =",
                      str(datetime.fromordinal(date + 1))
                      .split()[0])
        y_data = self.df_indo['new_cases'].values
        print('RMSE (root mean squared error)',
              mean_squared_error(y_data, new_case[:len(y_data)]))
        print('R^2 (coefficient of determination)',
              r2_score(y_data, new_case[:len(y_data)]))
        # Return
        return fig, ax

    def plot_observed_and_expected_new_case(self, pred_d, save):
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
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
    print()
    m.plot_observed_and_expected_new_case(90, save)
