from datetime import datetime

from weather_analytics import WeatherAnalytics

wa = WeatherAnalytics(history_from='2022-01-01')

today = datetime.utcnow().strftime('%Y-%m-%d')

dates = wa.find_top_matches_from_date(today, top_k=2)

best_dates, _ = wa.find_best_match()

wa.euclidean_distance(best_dates[0], best_dates[1], show_graph=True)

for date in dates:
    wa.euclidean_distance(today, date[1], show_graph=True)
