
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure import modelStructure
import pandas as pd

df_west_m = pd.read_csv("ScenarioAnalysis/Flights/flight_schedules_westminster.csv")


fl_id_to_type = dict(zip(df_west_m.nid, df_west_m.aircraft_type))
cost_funs = CostFuns()


def make_flight(line, slot_times, df_costs):
    slot_index = line["slot"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']
    fl_type = None

    # slot = Slot(slot_index, slot_time)
    if df_costs is None:

        delay_cost_vect, fl_id = cost_funs.get_random_cost_vect(slot_times, eta)
        fl_type = fl_id_to_type[fl_id]
    else:
        delay_cost_vect = df_costs[flight_name]

    print(fl_id, fl_type)

    return modelStructure.make_slot_and_flight(slot_time=slot_time, slot_index=slot_index, eta=eta,
                                               flight_name=flight_name, airline_name=airline_name,
                                               delay_cost_vect=delay_cost_vect, fl_type=fl_type)


def make_flight_list(df: pd.DataFrame, df_costs: pd.DataFrame = None):
    slot_list = []
    flight_list = []
    slot_times = df.time.to_numpy()
    for i in range(df.shape[0]):
        line = df.iloc[i]
        slot, flight = make_flight(line, slot_times, df_costs)
        slot_list.append(slot)
        flight_list.append(flight)

    return slot_list, flight_list


