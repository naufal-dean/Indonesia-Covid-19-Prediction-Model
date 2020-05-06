import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_data_1(df_indo, save):
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot line data
    ax.plot(df_indo['date'], df_indo['total_cases'], color='blue', linewidth=4)
    ax.plot(df_indo['date'], df_indo['total_deaths'], color='red', linewidth=4)
    # Setup axes format
    ax.set(
        xlabel='Tanggal',
        ylabel='Jumlah'
    )
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    # Setup figure
    fig.autofmt_xdate()
    fig.suptitle('Total Kasus Positif dan Meninggal Covid-19 di Indonesia\n'
                 'Hingga Tanggal 5 Mei 2020', fontsize=20)
    # Setup legend
    blue_line = mlines.Line2D([], [], color='blue', label='Positif')
    red_line = mlines.Line2D([], [], color='red', label='Meninggal')
    plt.legend(handles=[blue_line, red_line])
    # Show
    if (save):
        fig.savefig('plt/plot_total_positif_dan_meninggal.jpg')
    else:
        plt.show()


def plot_data_2(df_indo, y_bar_data_name, color, save):
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot line data
    ax.bar(df_indo['date'].values, df_indo[y_bar_data_name], color=color)
    # Setup axes format
    ax.set(
        xlabel='Tanggal',
        ylabel='Jumlah'
    )
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    # Setup figure
    fig.autofmt_xdate()
    type = 'Positif' if y_bar_data_name == 'new_cases' else 'Meninggal'
    fig.suptitle(f'Kasus {type} Baru Covid-19 di Indonesia\n'
                 'Hingga Tanggal 5 Mei 2020', fontsize=20)
    # Show
    if (save):
        fig.savefig(f'plt/plot_{y_bar_data_name}.jpg')
    else:
        plt.show()
