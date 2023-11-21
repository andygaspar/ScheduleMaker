import os

import pandas as pd

flights = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/flights_complete.csv"))
df_curfew = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/curfew.csv"))


def get_curfew_threshold(airport, airline, air_cluster, eta, min_turnaround):
    df_fl = flights[(flights.airport == airport) & (flights.airline == airline)
                    & (flights.aircraft_cluster == air_cluster)
                    & (eta - 60 <= flights.arr_min) & (flights.arr_min <= eta + 60)]

    if df_fl.shape[0] == 0:
        return None, None

    fl_random = df_fl.sample(1)
    registration = fl_random.registration.iloc[0]
    day = fl_random.arr_day.iloc[0]
    arr_time = fl_random.arr_min.iloc[0]

    df_rotation = flights[(flights.registration == registration) & (flights.arr_day == day) &
                          (flights.arr_min >= arr_time)].sort_values(by="arr_min")

    if df_rotation.shape[0] == 0:
        return None, None

    final_destination = df_rotation.iloc[-1].airport

    if final_destination not in df_curfew.Airport.to_list():
        return None, None

    flight_durations = (df_rotation.arr_min - df_rotation.dep_min).to_numpy()

    curfew_time = df_curfew[df_curfew.Airport == final_destination].CloseHour.iloc[0]
    open_curfew_time = df_curfew[df_curfew.Airport == final_destination].OpenHour.iloc[0]

    if flight_durations.shape[0] > 1:

        for t in flight_durations[1:]:
            curfew_time += -t - min_turnaround

    if open_curfew_time > curfew_time:
        curfew_time = 1440 + curfew_time

    return curfew_time - eta, final_destination
