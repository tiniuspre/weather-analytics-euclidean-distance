import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import requests
from sklearn.metrics import euclidean_distances
from dtaidistance import dtw_visualisation as dtwvis, dtw
from tqdm import tqdm


class WeatherAnalytics:
    def __init__(self, history_from='2022-01-01', history_to=None, lon=11.9977, lat=60.1905):
        self.lon = lon
        self.lat = lat
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        if history_to is None:
            history_to = (datetime.datetime.utcnow() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
        raw_today = self.get_raw_data('forecast', today, today)

        raw_history = self.get_raw_data('archive', history_from, history_to)
        self.data = self.sort_data(raw_history + raw_today)

    def get_raw_data(self, data_type, start_date, end_date):
        """forecast or archive as data type"""
        args = ''
        subdomain = ''
        if data_type == 'archive':
            args = f'&start_date={start_date}&end_date={end_date}'
            subdomain = 'archive-api'
        if data_type == 'forecast':
            args = '&forecast_days=1'
            subdomain = 'api'

        r = requests.get(
            f'https://{subdomain}.open-meteo.com/v1/{data_type}?latitude={self.lat}&longitude={self.lon}&hourly=temperature_2m' + args)

        res = r.json()['hourly']

        data = []
        day_info = {'time': [], 'temperature_2m': []}
        last_date = start_date

        for count in range(len(res['time'])):
            if last_date != res['time'][count][0:10]:
                data.append(day_info)
                day_info = {'time': [], 'temperature_2m': []}

            day_info['time'].append(res['time'][count])
            day_info['temperature_2m'].append(res['temperature_2m'][count])
            last_date = res['time'][count][0:10]

        data.append(day_info)

        return data

    def sort_data(self, data) -> dict[str, dict[str, int]]:
        sorted_data = {}
        for date_data in data:
            date = datetime.datetime.strptime(date_data["time"][0], '%Y-%m-%dT%H:%M')
            date_data.update(
                {"time": [datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M').hour for date_str in
                          date_data['time']]})
            sorted_data.update({date.strftime('%Y-%m-%d'): date_data})
        return sorted_data

    def display_date_graph(self, date: str) -> None:
        data = self.data[date]

        fig, ax = plt.subplots()
        ax.plot(data["time"], data["temperature_2m"])

        ax.set(xlabel='Time (H)', ylabel='Celsius (C)',
               title=f'{date} @ {data["time"][0]}h -> {data["time"][len(data["time"]) - 1]}h')
        ax.grid()
        plt.show()

    def merge_graphs(self, dates: list[str], text: str = 'Dates: ') -> None:
        fig, ax = plt.subplots()

        ax.set_prop_cycle(color=['r', 'g'])
        ax.text(0.99, 0.01, f'lon: {self.lon}, lat: {self.lat}',
                verticalalignment='bottom',
                horizontalalignment='right',
                weight='light',
                transform=ax.transAxes)

        for date in dates:
            date_data = self.data[date]
            ax.plot(date_data["time"], date_data["temperature_2m"], marker='o', ms=4)

        ax.set(xlabel='Time (H)', ylabel='Celsius (C)',
               title=f'{text} @ {" & ".join(dates)}')
        ax.legend(dates)
        ax.grid()

        plt.show()

    def euclidean_distance(self, date_1: str, date_2: str, show_graph=False) -> list[list[float]]:
        date_1_data = self.data[date_1]
        date_2_data = self.data[date_2]
        score = euclidean_distances([date_1_data['temperature_2m']], [date_2_data['temperature_2m']])

        if show_graph:
            self.merge_graphs([date_1, date_2], text=f'Score: {score[0][0]}')

        return score

    def graph_distance(self, date_1: str, date_2: str, show_graph=False) -> list[tuple]:
        date_1_data = self.data[date_1]
        date_2_data = self.data[date_2]

        fig, ax = plt.subplots(2, 1, figsize=(1280 / 96, 720 / 96))
        path = dtw.warping_path(date_1_data['temperature_2m'], date_2_data['temperature_2m'], window=24)
        dtwvis.plot_warping(date_1_data['temperature_2m'], date_2_data['temperature_2m'], path,
                            fig=fig, axs=ax)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title(f'DTW Warping Path Between {date_1} {date_2}')
        fig.tight_layout()
        if show_graph:
            plt.show()
        return path

    def find_best_match(self) -> (tuple[str, str], list[list[float]]):
        min_score = float('inf')
        best_pair = (None, None)
        total_length = sum(1 for _ in combinations(self.data.keys(), 2))

        for date_1, date_2 in tqdm(combinations(self.data.keys(), 2), total=total_length):
            score = self.euclidean_distance(date_1, date_2)
            if score < min_score:
                min_score = score
                best_pair = (date_1, date_2)

        return best_pair, min_score

    def find_top_matches_from_date(self, date: str, top_k: int = 5) -> list[tuple[list[list[float]], str]]:
        scores_dates = []
        for match_date in self.data.keys():
            if match_date == date:
                continue
            score = self.euclidean_distance(date, match_date)
            scores_dates.append((score, match_date))

        scores_dates.sort(key=lambda x: x[0])

        top_matches = scores_dates[:top_k]
        return top_matches
