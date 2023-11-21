import time
from matplotlib import pyplot as plt

from SchedulePackage.ScheduleGenerator.real_schedule import RealisticSchedule as Rs
from SchedulePackage.ScheduleGenerator.real_schedule_parallel import RealisticScheduleParallel as RsPar

sm = Rs()
schedule_maker = RsPar()
regulation = schedule_maker.get_regulation()
regulation.nFlights = 200

t = time.time()
slot_list, fl_list = schedule_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                         n_flights=regulation.nFlights,
                                                         capacity_reduction=regulation.cReduction,
                                                         regulation_time=regulation.startTime)

print(time.time() - t)

for f in fl_list[:20]:
    plt.plot(f.delay_cost_vect)
    print(f.missed_connected)
plt.show()


# t = time.time()
# slot_list_seq, fl_list_seq = sm.make_sl_fl_from_data(airport=regulation.airport,
#                                                      n_flights=regulation.nFlights,
#                                                      capacity_reduction=regulation.cReduction,
#                                                      regulation_time=regulation.startTime)
#
# print(time.time() - t)
