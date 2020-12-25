#!/usr/bin/env python3

import math
import datetime
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
import requests
from bs4 import BeautifulSoup
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyimgur

matplotlib.use('Agg')


def parse_covid_data(filename):
    "Loads the date from file into two lists separate x and y list"
    with open(filename) as file:
        content = file.readlines()

    x_data = []
    y_data = []
    last_date = ''
    base_date = ''

    for line in content:
        fields = line.split()
        if len(fields) < 2:
            continue
        parsed_date = datetime.datetime.strptime(fields[0], '%Y-%m-%d')
        if base_date == '':
            base_date = parsed_date
        last_date = fields[0]
        date = (parsed_date - base_date).days
        number = float(fields[1])
        x_data.append(date)
        y_data.append(number)

    return {
        'x_data': x_data,
        'y_data': y_data,
        'base_date': base_date,
        'last_date_str': last_date
    }


def get_logistic_model(y_base):
    "Generates logistic model function for the given Y base"

    def logistic_model(day, x_scale, peak, max_cases):
        "Logistic model formula"
        return max_cases / (1 + np.exp(-(day - peak) / x_scale)) + y_base

    return logistic_model


def fit_logistic_model(x_data, y_data, base_date):
    "Fits data into logistic curve"

    try:
        sigma = [1] * len(y_data)
        # sigma[-1] = 0.1
        model = get_logistic_model(y_data[0])
        popt, pcov = curve_fit(model, x_data, y_data, p0=[
            2, 60, 100000], sigma=sigma)
        errors = np.sqrt(np.diag(pcov))
        peak_date = (base_date + datetime.timedelta(days=popt[1]))
        peak_date_error = errors[1]
        max_inf = popt[2]
        max_inf_error = errors[2]

        if peak_date_error > 1e7 or max_inf_error > 1e7:
            print("No sigmoid fit due to too large covariance.")
            return None

        if max_inf_error > max_inf:
            print(
                "No sigmoid fit because the uncertainty of the "
                "maximum is larger than the maximum itself.")
            return None

        return {
            'peak': popt[1],
            'peak_date': peak_date,
            'peak_date_error': peak_date_error,
            'peak_growth': model(popt[1] + 1, popt[0], popt[1], popt[2])
                           - model(popt[1], popt[0], popt[1], popt[2]),
            'tomorrow_growth':
                model(x_data[-1] + 1, popt[0], popt[1], popt[2]) - y_data[-1],
            'max_inf': max_inf,
            'max_inf_error': max_inf_error,
            'x_scale': popt[0],
            'x_scale_error': errors[0],
            'popt': popt,
            'pcov': pcov
        }
    except RuntimeError as rte:
        print("No sigmoid fit due to exception: {}".format(rte))
        return None


def get_exponential_model(y_base):
    "Generates exponential model function for the given Y base"

    def exponential_model(day, ln_daily_growth, x_shift):
        "Exponential model formula"

        return np.exp(ln_daily_growth * (day - x_shift)) + y_base

    return exponential_model


def fit_exponential_model(x_data, y_data):
    "Fits exponential model to data"
    sigma = [1] * len(y_data)
    # sigma[-1] = 0.1
    model = get_exponential_model(y_data[0])
    popt, pcov = curve_fit(model, x_data, y_data, sigma=sigma, maxfev=5000)
    params = popt
    errors = np.sqrt(np.diag(pcov))

    return {
        'ln_daily_growth': params[0],
        'ln_daily_growth_error': errors[0],
        'daily_growth': np.exp(params[0] + errors[0] ** 2 / 2),
        'tomorrow_growth': model(x_data[-1] + 1, popt[0], popt[1]) - y_data[-1],
        'raw_daily_growth': np.exp(params[0]),
        'daily_growth_error': np.sqrt(
            (np.exp(errors[0] ** 2) - 1) *
            np.exp(2 * params[0] + errors[0] ** 2)
        ),
        'x_shift': params[1],
        'x_shift_error': errors[1],
        'popt': popt,
        'pcov': pcov
    }


def create_curve_data(x_data, y_data, base_date, log_result, exp_result):
    """
    Creates the curves to be used when plotting data based
    on the calculated results.
    """
    if log_result is None:
        days_to_simulate = 2 * (x_data[-1] - x_data[0] + 1)
    else:
        days_to_simulate = max(
            2 * (log_result['peak_date'] - base_date).days,
            x_data[-1] - x_data[0] + 1
        )

    days = range(x_data[0], x_data[0] + days_to_simulate)
    out_date = [base_date + datetime.timedelta(days=x)
                for x in range(x_data[0], x_data[0] + days_to_simulate)]

    out_y = y_data + [float('nan')] * (days_to_simulate - len(y_data))

    if log_result is not None:
        out_log = [get_logistic_model(y_data[0])(
            x, *log_result['popt']) for x in days]
    else:
        out_log = [float('nan')] * days_to_simulate

    out_exp = [get_exponential_model(y_data[0])(
        x, *exp_result['popt']) for x in days]

    return {
        'date': out_date,
        'y': out_y,
        'logistic': out_log,
        'exponential': out_exp
    }


def print_curves(curve_data):
    "Prints the curve data into terminal."

    print("{:<15}{:<15}{:<15}{:<15}".format(
        "Date", "Actual", "Predicted log", "Predicted exp"))
    for i in range(0, len(curve_data['date'])):
        print("{:<15}{:>15}{:>15.2f}{:>15.2f}".format(
            curve_data['date'][i].strftime('%Y-%m-%d'),
            curve_data['y'][i],
            curve_data['logistic'][i],
            curve_data['exponential'][i]
        ))


def save_plot(curve_data, covid_data, log_result, texts):
    "Generates and saves the plot."

    axes = plt.gca()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    axes.xaxis.set_minor_locator(mdates.DayLocator())

    plt.figure(figsize=[10.24, 7.68])
    plt.plot(curve_data['date'], curve_data['y'],
             texts['element_marker'], label=texts['cases_axis_name'])
    if log_result is not None:
        plt.plot(curve_data['date'], curve_data['logistic'],
                 'g-', label='Sigmoid model')
    plt.plot(curve_data['date'], curve_data['exponential'],
             'b-', label='Exponential model')
    plt.ylabel(texts['y_axis_name'])
    plt.xlabel('Date')
    if log_result is None:
        max_y = 2 * max(covid_data['y_data'])
    else:
        max_y = max(curve_data['logistic'] + covid_data['y_data'])
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   texts['max_inf_str'] + "\n" +
                   texts['peak_date_str'] + "\n" +
                   texts['daily_growth_str'], va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(
        curve_data['date']), covid_data['y_data'][0], max_y])
    plt.legend()
    plt.grid()
    plt.title("{} {}".format(texts['plot_title'], covid_data['last_date_str']))
    file_name = 'plot' + texts['plot_file_suffix'] + '.png'
    plt.savefig(file_name)
    print("Plot saved to {}".format(file_name))


def create_plots(texts):
    # x_data, y_data, base_date, last_date
    covid_data = parse_covid_data(texts['file_name'])

    log_result = fit_logistic_model(
        covid_data['x_data'], covid_data['y_data'], covid_data['base_date'])
    if log_result is not None:
        texts['peak_date_str'] = (
            "Sigmoid inflection point: "
            "{} ± {:.2f} nap"
            " (Maximum slope: {:.2f}, f(x+1) - y(x) ≈ {:.2f})").format(
            log_result['peak_date'].strftime(
                '%Y-%m-%d'), log_result['peak_date_error'],
            log_result['peak_growth'], log_result['tomorrow_growth']
        )
        texts['max_inf_str'] = "Sigmoid maximum: {:.2f} ± {:.2f}".format(
            log_result['max_inf'] + covid_data['y_data'][0],
            log_result['max_inf_error']
        )
        print(texts['max_inf_str'])
    else:
        texts['peak_date_str'] = "The Sigmoid model does not fit the data."
        texts['max_inf_str'] = ""
        print("Logistic curve is too bad fit for current data")

    exp_result = fit_exponential_model(
        covid_data['x_data'], covid_data['y_data'])
    print(exp_result)
    texts['daily_growth_str'] = (
        "Daily growth based on the exponential model:"
        " {:.2f}% ± {:.2}%."
        " (Duplázódás: {:.2f} naponta, f(x+1) - y(x) ≈ {:.2f})").format(
        exp_result['daily_growth'] * 100 - 100, exp_result['daily_growth_error'] *
        100, math.log(
            2) / math.log(exp_result['daily_growth']),
        exp_result['tomorrow_growth']
    )
    print(texts['daily_growth_str'])
    print("ln daily growth: {}, x_shift: {}".format(
        exp_result["ln_daily_growth"], exp_result["x_shift"]))

    curve_data = create_curve_data(
        covid_data['x_data'],
        covid_data['y_data'],
        covid_data['base_date'],
        log_result,
        exp_result
    )
    # print_curves(curve_data)
    save_plot(curve_data, covid_data, log_result, texts)


def month_translator(month):
    switch = {
        "január": "01",
        "február": "02",
        "március": "03",
        "április": "04",
        "május": "05",
        "június": "06",
        "július": "07",
        "augusztus": "08",
        "szeptember": "09",
        "október": "10",
        "november": "11",
        "december": "12"
    }
    return switch.get(month, "Invalid day of month")


def read_covid_data():
    with open('covid_data.txt', 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        f.close()
    with open('covid_deaths.txt', 'r') as f:
        lines = f.read().splitlines()
        last_deaths_line = lines[-1]
        f.close()
    return last_line, last_deaths_line


def scrape(last_date):
    print("Scrapping started from today until " + str(last_date))
    max_pages_to_scan = 50
    data = []
    for page in range(max_pages_to_scan):
        url = 'https://koronavirus.gov.hu/hirek?page=' + str(page)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.find_all(class_='article-teaser')
        for article in articles:
            title = article.find('h3').text.strip().split()
            if len(title) > 6 and title[1] == "fővel" and title[2] == "emelkedett":
                infected = title[0]
            elif len(title) > 6 and title[2] == "fővel" and title[3] == "emelkedett": # 4 405 fővel emelkedet..
                infected = title[0] + title[1]
            else:
                continue
            for i in range(6, len(title)):  # krónikus or idős, sometimes both, sometimes neither..
                if title[i] == "elhunyt":
                    death = title[i + 1]
                    break
            date = article.find('i').text.strip()[:-9].replace(".", "")
            date_s = date.split()
            date = date_s[0] + "-" + month_translator(date_s[1]) + "-" + date_s[2]
            if datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d') > last_date:
                triplet = (date, infected, death)
                data.append(triplet)
                print(triplet)
            else:
                return data
    print("Could not scrap back until " + str(last_date))
    print("Last scrapped date: " + str(data[-1][0]))
    return data


def update_data():
    last_line, last_deaths_line = read_covid_data()
    last_date = datetime.datetime.strptime(last_line[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    if today == last_date:
        print("Already scrapped today's data.")
        return
    data = scrape(last_date)
    data.reverse()
    if len(data) == 0:
        print("Could not scrap data from web. The data for today is probably not out yet.")
        return
    covid_string = ""
    deaths_string = ""
    total_cases = int(last_line.split()[1])
    total_deaths = int(last_deaths_line.split()[1])
    for i in range(len(data)):
        total_cases += int(data[i][1])
        total_deaths += int(data[i][2])
        covid_string += data[i][0] + " " + str(total_cases) + " +" + str(data[i][1]) + "\n"
        deaths_string += data[i][0] + " " + str(total_deaths) + " +" + str(data[i][2]) + "\n"
    covid_string = covid_string.rstrip('\n')
    deaths_string = deaths_string.rstrip('\n')
    print(covid_string)
    print(deaths_string)
    with open("covid_data.txt", "a") as f:
        f.write('\n' + covid_string)
    with open("covid_deaths.txt", "a") as f:
        f.write('\n' + deaths_string)


def upload_images():
    with open('imgurcreds.txt') as f:
        client_id = f.readline().rstrip('\n')
    plot_path = "plot.png"
    plot_deaths_path = "plot-deaths.png"
    im = pyimgur.Imgur(client_id)
    plot = im.upload_image(plot_path, title="Covid-19 graph")
    plot_deaths = im.upload_image(plot_deaths_path, title="Covid-19 deaths")
    print(plot.link)
    print(plot_deaths.link)
    return plot.link, plot_deaths.link


def edit_readme(links):
    with open('README.md', 'r') as file:
        data = file.readlines()
    data[3] = "![Covid curve image](" + str(links[0]) + ")\n"
    data[4] = "![Covid curve deaths image](" + str(links[1]) + ")\n"
    with open('README.md', 'w') as file:
        file.writelines(data)
    print("Updated README.md")


def main():
    update_data()
    texts = {
        'file_name': 'covid_deaths.txt',
        'cases_axis_name': 'Total deaths',
        'y_axis_name': 'Total deaths',
        'element_marker': 'k+',
        'plot_file_suffix': '-deaths',
        'plot_title': 'COVID-19 curve fitting - total deaths',
    }
    create_plots(texts)
    texts = {
        'file_name': 'covid_data.txt',
        'cases_axis_name': 'Reported cases',
        'y_axis_name': 'Total cases',
        'element_marker': 'ro',
        'plot_file_suffix': '',
        'plot_title': 'COVID-19 curve fitting - total cases',
    }
    create_plots(texts)
    links = upload_images()
    edit_readme(links)


if __name__ == "__main__":
    main()
