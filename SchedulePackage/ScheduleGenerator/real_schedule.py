import os

import numpy as np
import pandas as pd
from CostPackage.arrival_costs import get_cost_model, get_data_dict

from SchedulePackage.ScheduleGenerator.flight_and_regulation import Flight, Regulation
from SchedulePackage.Utils.curfew import get_curfew_threshold
from SchedulePackage.Utils.flight_length import get_flight_length


class RealisticSchedule:

    def __init__(self):
        self.df_airline = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                   "../SchedulesData/airport_airline_frequency.csv"))

        self.low_costs = self.df_airline[self.df_airline.low_cost == True].airline.unique().tolist()

        self.df_aircraft_high = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                         "../SchedulesData/aircraft_high.csv"))
        self.df_aircraft_low = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                        "../SchedulesData/aircraft_low.csv"))
        self.aircraft_seats = get_data_dict()["aircraft_seats"]
        self.df_capacity = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/airport_max_capacity"
                                                                               ".csv"))
        self.pax = pd.read_csv(os.path.join(os.path.dirname(__file__), "../SchedulesData/pax.csv"))

        # iata increase load factor
        self.pax.pax = self.pax.pax.apply(lambda pp: int(pp + pp * 0.021))
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
                             regulation_time: int = 0, compute=True):

        df_airline = self.df_airline[self.df_airline.airport == airport]
        capacity = self.df_capacity[self.df_capacity.airport == airport].capacity.iloc[0]

        interval = 60 / capacity
        new_interval = 60 / (capacity * (1 - capacity_reduction))
        times = np.linspace(0, n_flights * interval, n_flights)
        new_times = np.linspace(0, n_flights * new_interval, n_flights)

        df_airport = self.df_airport_airline_aircraft[self.df_airport_airline_aircraft.airport == airport]

        flight_list = []
        slot_list = [{'index': i, 'initial_time': times[i], 'new_time': new_times[i]} for i in range(times.shape[0])]

        for i in range(n_flights):
            airline = df_airline.airline.sample(weights=df_airline.frequency).iloc[0]
            is_low_cost = airline in self.low_costs
            df_airport_airline = df_airport[df_airport.airline == airline]
            fl_type = df_airport_airline.air_cluster.sample(weights=df_airport_airline.frequency).iloc[0]

            passengers = self.get_passengers(airport=airport, airline=airline,
                                             air_cluster=fl_type, load_factor=load_factor)
            missed_connected = self.get_missed_connected(airport=airport, airline=airline, passengers=passengers)

            length = get_flight_length(airline=airline, airport=airport, air_cluster=fl_type)

            eta = regulation_time + times[i]
            min_turnaround = self.turnaround_dict[fl_type]
            curfew_th, rotation_destination = get_curfew_threshold(airport, airline, fl_type, eta, min_turnaround)
            curfew = (curfew_th, self.get_passengers(rotation_destination, airline, fl_type, load_factor)) \
                if curfew_th is not None else None

            cost_fun = get_cost_model(aircraft_type=fl_type, is_low_cost=is_low_cost, destination=airport,
                                      length=length, n_passengers=passengers, missed_connected=missed_connected,
                                      curfew=curfew)

            delay_cost_vect = np.array([cost_fun(new_times[j]) for j in range(n_flights)])

            curfew = curfew[0] if curfew is not None else curfew
            flight_list.append(
                Flight(i, times[i], new_times[i], airline, is_low_cost, fl_type, passengers, missed_connected, curfew, length,
                       cost_fun, delay_cost_vect))

        return slot_list, flight_list

    def get_regulation(self, capacity_min=0., n_flights_min=0, n_flights_max=1000, start=0, end=1441):
        regulations = self.regulations[(self.regulations.capacity_reduction_mean >= capacity_min) &
                                       (self.regulations.n_flights >= n_flights_min) &
                                       (self.regulations.n_flights <= n_flights_max) &
                                       (self.regulations.min_start >= start) &
                                       (self.regulations.min_end <= end)]
        regulation = regulations.sample().iloc[0]
        regulation = Regulation(airport=regulation.ReferenceLocationName, n_flights=regulation.n_flights,
                                c_reduction=regulation.capacity_reduction_mean,
                                start_time=regulation.min_start)
        return regulation

    def get_passengers(self, airport, airline, air_cluster, load_factor):
        pax = self.pax[(self.pax.destination == airport)
                       & (self.pax.airline == airline)
                       & (self.pax.air_cluster == air_cluster)]
        if pax.shape[0] > 0:
            flight_sample = pax.leg1.sample().iloc[0]
            passengers = pax[pax.leg1 == flight_sample].pax.sum()
        else:
            passengers = int(self.aircraft_seats[self.aircraft_seats.Aircraft == air_cluster]["SeatsLow"].iloc[0]
                             * load_factor)
        return passengers

    def get_missed_connected(self, airport, airline, passengers):
        pax_connections = self.pax[(self.pax.destination == airport) & (self.pax.airline == airline)]
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
