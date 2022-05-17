# -*- coding: utf-8 -*-
"""
Creating power demand profiles using bdew profiles.

Installation requirements
-------------------------
This example requires at least version v0.1.4 of the oemof demandlib. Install
by:
    pip install 'demandlib>=0.1.4,<0.2'
Optional:
    pip install matplotlib

SPDX-FileCopyrightText: Birgit Schachler
SPDX-FileCopyrightText: Uwe Krien <krien@uni-bremen.de>
SPDX-FileCopyrightText: Stephen Bosch

SPDX-License-Identifier: MIT

"""
import pandas as pd
from datetime import datetime
from datetime import time as settime

import numpy as np
from matplotlib import pyplot as plt

import demandlib.bdew as bdew
import demandlib.particular_profiles as profiles


# The following dictionary is create by "workalendar"
# pip3 install workalendar
# >>> from workalendar.europe import Germany
# >>> cal = Germany()
# >>> holidays = dict(cal.holidays(2010))

holidays = {
    datetime(2016, 5, 16): 'Whit Monday',
    datetime(2016, 3, 28): 'Easter Monday',
    datetime(2016, 5, 5): 'Ascension Thursday',
    datetime(2016, 1, 1): 'New year',
    datetime(2016, 10, 3): 'Day of German Unity',
    datetime(2016, 12, 25): 'Christmas Day',
    datetime(2016, 5, 1): 'Labour Day',
    datetime(2016, 3, 25): 'Good Friday',
    datetime(2016, 12, 26): 'Second Christmas Day'
}
# create dataframe for 2016
demand = pd.DataFrame(
    index=pd.date_range(datetime(2016, 1, 1, 0),
                        periods=8784, freq='H'))

#Typ 	Beschreibung 	Erläuterung
# G0 	Gewerbe allgemein             	Gewogener Mittelwert der Profile G1-G6
# G1 	Gewerbe werktags 8–18 Uhr     	z.B. Büros, Arztpraxen, Werkstätten, Verwaltungseinrichtungen
# G2 	Gewerbe mit starkem bis überwiegendem Verbrauch in den Abendstunden 	z.B. Sportvereine, Fitnessstudios, Abendgaststätten
# G3 	Gewerbe durchlaufend 	z.B. Kühlhäuser, Pumpen, Kläranlagen
# G4 	Laden/Friseur 	 
# G5 	Bäckerei mit Backstube 	 
# G6 	Wochenendbetrieb 	z.B. Kinos
# G7 	Mobilfunksendestation 	durchgängiges Bandlastprofil
# L0 	Landwirtschaftsbetriebe allgemein 	Gewogener Mittelwert der Profile L1 und L2
# L1 	Landwirtschaftsbetriebe mit Milchwirtschaft/Nebenerwerbs-Tierzucht 	 
# L2 	Übrige Landwirtschaftsbetriebe 	 
# H0/H0_dyn 	Haushalt/Haushalt dynamisiert
ann_el_demand_per_sector = {
    "g0": 10800,
    "h0": 410489,
    # "h0_dyn": 416564,
    # "i0": 3000,
    # "i1": 5000,
    # "i2": 6000,
    "g1": 51255,
    "g2": 22120,
    "g5": 33000,
}

year = 2016

# read standard load profiles
e_slp = bdew.ElecSlp(year, holidays=holidays)

# multiply given annual demand with timeseries
elec_demand = e_slp.get_profile(ann_el_demand_per_sector)

# Add the slp for the industrial group
# ilp = profiles.IndustrialLoadProfile(e_slp.date_time_index, holidays=holidays)

# Beginning and end of workday, weekdays and weekend days, and scaling
# factors by default
# elec_demand["i0"] = ilp.simple_profile(ann_el_demand_per_sector["i0"])

# # Set beginning of workday to 9 am
# elec_demand["i1"] = ilp.simple_profile(
#     ann_el_demand_per_sector["i1"], am=settime(9, 0, 0)
# )

# # Change scaling factors
# elec_demand["i2"] = ilp.simple_profile(
#     ann_el_demand_per_sector["i2"],
#     profile_factors={
#         "week": {"day": 1.0, "night": 0.8},
#         "weekend": {"day": 0.8, "night": 0.6},
#     },
# )

print(
    "Be aware that the values in the DataFrame are 15 minute values"
    + "with a power unit. If you sum up a table with 15min values"
    + "the result will be of the unit 'kW15minutes'."
)
print(elec_demand.sum())

print("You will have to divide the result by 4 to get kWh.")
print(elec_demand.sum() / 4)

print("Or resample the DataFrame to hourly values using the mean() " "method.")

# Resample 15-minute values to hourly values.
elec_demand_resampled = elec_demand.resample("H").mean()
print(elec_demand_resampled.sum())
demand = elec_demand_resampled

demand.to_csv('in/el_demand_bdew.csv')

# Plot demand
ax = elec_demand_resampled.plot()
ax.set_xlabel("Date")
ax.set_ylabel("Power demand")
plt.show()

print(elec_demand)

for key in ann_el_demand_per_sector:
    assert np.isclose(
        elec_demand[key].sum() / 4, ann_el_demand_per_sector[key]
    )
