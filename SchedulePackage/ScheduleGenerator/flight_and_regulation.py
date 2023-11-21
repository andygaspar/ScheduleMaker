import numpy as np

class Flight:

    def __init__(self, idx, fl_time, new_time, airline, is_low_cost, fl_type, passengers, missed_connected, curfew, length,
                 cost_fun, delay_cost_vect):
        self.slot_index = idx
        self.eta = fl_time
        self.slot_time = new_time
        self.flight_name = airline + str(idx)
        self.airline_name = airline
        self.is_low_cost = is_low_cost
        self.fl_type = fl_type
        self.passengers = passengers
        self.missed_connected = missed_connected
        self.curfew = curfew
        self.length = length
        self.cost_fun = cost_fun
        self.delay_cost_vect = delay_cost_vect


class Regulation:
    def __init__(self, airport, n_flights, c_reduction, start_time):
        self.airport = airport
        self.nFlights = n_flights
        c_reduction = np.around(c_reduction, decimals=1)
        self.cReduction = c_reduction if c_reduction >= 0.1 else 0.1
        self.startTime = start_time

