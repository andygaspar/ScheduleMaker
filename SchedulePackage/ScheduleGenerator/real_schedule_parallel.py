import os
from typing import List

import numpy as np
import pandas as pd
from CostPackage.arrival_costs import get_cost_model, get_data_dict
import multiprocessing as mp

df_curfew = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/curfew.csv"))

from SchedulePackage.ScheduleGenerator.flight_and_regulation import ScheduleFlight, Regulation, ScheduleSlot

flights = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/flights_complete.csv"))

pax = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/pax.csv"))
# iata increase load factor
pax.pax = pax.pax.apply(lambda pp: int(pp + pp * 0.021))
aircraft_seats = get_data_dict()["aircraft_seats"]


def get_passengers(airport, airline, air_cluster, load_factor):
    passengers = pax[(pax.destination == airport)
                     & (pax.airline == airline)
                     & (pax.air_cluster == air_cluster)]
    if passengers.shape[0] > 0:
        flight_sample = pax.leg1.sample().iloc[0]
        passengers = pax[pax.leg1 == flight_sample].pax.sum()
    else:
        passengers = int(aircraft_seats[aircraft_seats.Aircraft == air_cluster]["SeatsLow"].iloc[0]
                         * load_factor)
    return passengers


def get_missed_connected(airport, airline, passengers):
    pax_connections = pax[(pax.destination == airport) & (pax.airline == airline)]
    if pax_connections.shape[0] > 0:
        pax_connections = pax_connections.sample(n=passengers, weights=pax_connections.pax, replace=True)
        pax_connections = pax_connections[pax_connections.leg2 > 0]
        if pax_connections.shape[0] > 0:
            missed_connected = pax_connections.apply(lambda x: (x.delta_leg1, x.delay), axis=1).to_list()
        else:
            missed_connected = None
    else:
        missed_connected = None

    return missed_connected


def get_flight_length(airline: str, airport: str, air_cluster: str):
    df_flights = flights[(flights.airport == airport)
                         & (flights.airline == airline) & (flights.aircraft_cluster == air_cluster)]
    if df_flights.shape[0] == 0:
        print("flight not found")
        return 4000
    else:
        return df_flights.length.sample().iloc[0]


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


def make_flight(airport, airline, fl_type, eta, min_turnaround, load_factor, is_low_cost, idx, new_times, fl_time):
    passengers = get_passengers(airport=airport, airline=airline,
                                air_cluster=fl_type, load_factor=load_factor)
    missed_connected = get_missed_connected(airport=airport, airline=airline, passengers=passengers)

    length = get_flight_length(airline=airline, airport=airport, air_cluster=fl_type)

    curfew_th, rotation_destination = get_curfew_threshold(airport, airline, fl_type, eta, min_turnaround)
    curfew = (curfew_th, get_passengers(rotation_destination, airline, fl_type, load_factor)) \
        if curfew_th is not None else None

    curfew = curfew[0] if curfew is not None else curfew
    flight = ScheduleFlight(
        idx, fl_time, new_times[idx], airline, is_low_cost, fl_type, passengers, missed_connected, curfew, length,
        None, None)

    return flight


class RealisticScheduleParallel:

    def __init__(self):
        self.df_airline = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                   "../SchedulesData/airport_airline_frequency.csv"))

        self.low_costs = self.df_airline[self.df_airline.low_cost == True].airline.unique().tolist()

        self.df_aircraft_high = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                         "../SchedulesData/aircraft_high.csv"))
        self.df_aircraft_low = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                        "../SchedulesData/aircraft_low.csv"))
        self.df_capacity = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/airport_max_capacity"
                                                                               ".csv"))

        self.df_turnaround = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                      "../SchedulesData/turnaround.csv"))
        self.df_airport_airline_aircraft = \
            pd.read_csv(os.path.join(os.path.dirname(__file__),
                                     "../SchedulesData/airport_airline_cluster_frequency.csv"))

        self.turnaround_dict = dict(zip(self.df_turnaround.AirCluster, self.df_turnaround.MinTurnaround))

        self.regulations = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                    "../SchedulesData/regulations_25_nonzero.csv"))

    def get_rand_airport(self):
        return np.random.choice(self.df_airline.airport.to_list())

    def make_sl_fl_from_data(self, airport: str, n_flights: int, capacity_reduction: float, load_factor=0.89,
                             regulation_time: int = 0):

        df_airline = self.df_airline[self.df_airline.airport == airport]
        capacity = self.df_capacity[self.df_capacity.airport == airport].capacity.iloc[0]

        interval = 60 / capacity
        new_interval = 60 / (capacity * (1 - capacity_reduction))
        times = np.linspace(0, n_flights * interval, n_flights)
        new_times = np.linspace(0, n_flights * new_interval, n_flights)

        df_airport = self.df_airport_airline_aircraft[self.df_airport_airline_aircraft.airport == airport]

        slot_list = [ScheduleSlot(index=i, original_time=times[i], new_time=new_times[i]) for i in range(times.shape[0])]

        args = []
        for idx in range(n_flights):
            airline = df_airline.airline.sample(weights=df_airline.frequency).iloc[0]
            is_low_cost = airline in self.low_costs
            df_airport_airline = df_airport[df_airport.airline == airline]
            fl_type = df_airport_airline.air_cluster.sample(weights=df_airport_airline.frequency).iloc[0]

            eta = regulation_time + times[idx]
            min_turnaround = self.turnaround_dict[fl_type]

            args.append((airport, airline, fl_type, eta, min_turnaround, load_factor, is_low_cost, idx, new_times, times[idx]))

        with mp.Pool(mp.cpu_count()) as pool:
            flight_list = pool.starmap(make_flight, args)

        flight_list: List[ScheduleFlight]
        for fl in flight_list:
            cost_fun = get_cost_model(aircraft_type=fl.fl_type, is_low_cost=fl.is_low_cost, destination=airport,
                                      length=fl.length, n_passengers=fl.passengers, missed_connected=fl.missed_connected,
                                      curfew=fl.curfew)

            delay_cost_vect = np.array([cost_fun(t) for t in new_times])
            fl.cost_fun = cost_fun
            fl.delay_cost_vect = delay_cost_vect

        return slot_list, flight_list

    def get_regulation(self, capacity_min=0., n_flights_min=0, n_flights_max=1000, start=0, end=1441):
        regulations = self.regulations[(self.regulations.capacity_reduction_mean >= capacity_min) &
                                       (self.regulations.n_flights >= n_flights_min) &
                                       (self.regulations.n_flights <= n_flights_max) &
                                       (self.regulations.min_start >= start) &
                                       (self.regulations.min_end <= end)]
        regulation = regulations.sample().iloc[0]
        regulation = Regulation(airport=regulation.airport, n_flights=regulation.n_flights,
                                c_reduction=regulation.capacity_reduction_mean,
                                start_time=regulation.min_start)
        return regulation
