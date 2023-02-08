from typing import Union, List

import numpy as np
import random
import string
from scipy import stats
import pandas as pd
from SchedulePackage.ScenarioAnalysis.Aircraft import aircraft_analysis as aircraft


def avoid_zero(flight_list, num_flights):
    while len(flight_list[flight_list < 1]) > 0:
        for i in range(flight_list.shape[0]):
            if flight_list[i] == 0:
                flight_list[i] += 1
                if sum(flight_list) > num_flights:
                    flight_list[np.argmax(flight_list)] -= 1
    return flight_list


def fill_missing_flights(flight_list, num_flights, num_airlines):
    missing = num_flights - sum(flight_list)
    for i in range(missing):
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), flight_list / sum(flight_list)))
        flight_list[custm.rvs(size=1)] += 1
    return np.flip(np.sort(flight_list))


def distribution_maker(num_flights, num_airlines, distribution="uniform"):
    dist = []

    if distribution == "uniform":
        h, loc = np.histogram(np.random.uniform(0, 1, 1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_high_few_low":
        f = lambda x: x ** 3 + 1
        base = np.linspace(-1, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_low":
        f = lambda x: x ** 4 + 1
        base = np.linspace(-1, 1 / 4, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_high":
        f = lambda x: x ** 2
        base = np.linspace(0, 1, num_airlines)
        val = f(base)
        val[val > 3 / 4] = 3 / 4
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), val / sum(val)))
        h, l = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "increasing":
        f = lambda x: x
        base = np.linspace(0, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "hub":
        f = lambda x: x ** 10
        base = np.linspace(0, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    dist = avoid_zero(dist, num_flights)
    dist = fill_missing_flights(dist, num_flights, num_airlines)
    return dist


def df_maker(num_flights=20, num_airlines=3, distribution="uniform", capacity=1, new_capacity=2,
             custom: Union[None, List[int]] = None, aircraft_type="from_distribution"):

    if custom is None:
        dist = distribution_maker(num_flights, num_airlines, distribution)
        airline = [[string.ascii_uppercase[j] for i in range(dist[j])] for j in range(num_airlines)]
    else:
        num_airlines = len(custom)
        num_flights = sum(custom)
        airline = [[string.ascii_uppercase[j] for i in range(custom[j])] for j in range(num_airlines)]

    airline = [val for sublist in airline for val in sublist]
    airline = np.random.permutation(airline)
    flights = ["F" + airline[i] + str(i) for i in range(num_flights)]

    slot = np.arange(num_flights)
    eta = slot * capacity
    fpfs = slot * new_capacity
    priority = []
    for i in range(num_flights):
        m = random.choice([0, 1])
        if m == 0:
            priority.append(np.random.normal(0.7, 0.1))
        else:
            priority.append(np.random.normal(1.5, 0.1))

    priority = np.abs(priority)
    cost = priority
    num = range(num_flights)
    margins_gap = np.array([random.choice(range(15, 45)) for i in num])

    if aircraft_type == "at_gate":
        at_gate = pd.read_csv("ModelStructure/Costs/costs_table_gate.csv", sep=" ")
        flights_type = [np.random.choice(at_gate["flight"].to_numpy()) for i in range(num_flights)]
    else:
        flights_type = [aircraft.get_random_flight_type() for _ in range(num_flights)]

    return pd.DataFrame(
        {"slot": slot, "flight": flights, "eta": eta, "fpfs": fpfs, "time": fpfs, "priority": priority,
         "margins":margins_gap, "airline": airline, "cost": cost, "num": num, "type": flights_type})


def schedule_types(show=False):
    dfTypeList = ("uniform", "few_low", "few_high", "increasing", "hub")
    if show:
        print(dfTypeList)
    return dfTypeList
