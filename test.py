import pandas as pd
import os

from SchedulePackage.ScheduleGenerator.real_schedule import RealisticSchedule
schedule_maker = RealisticSchedule()
regulation = schedule_maker.get_regulation()
regulation.nFlights = 120

slot_list, fl_list = schedule_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                         n_flights=regulation.nFlights,
                                                         capacity_reduction=regulation.cReduction,
                                                         compute=True, regulation_time=regulation.startTime)

