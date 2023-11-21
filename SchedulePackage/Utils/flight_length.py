import os

import pandas as pd

flights = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/flights_complete.csv"))


def get_flight_length(airline: str, airport: str, air_cluster: str):
    df_flights = flights[(flights.airport == airport)
                         & (flights.airline == airline) & (flights.aircraft_cluster == air_cluster)]
    if df_flights.shape[0] == 0:
        print("flight not found")
        return 4000
    else:
        return df_flights.length.sample().iloc[0]
