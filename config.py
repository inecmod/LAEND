import pandas as pd
import os
import shutil
from datetime import datetime

import logging


#######################################
#------Model specific settings
#######################################
#----- Temporal Settings
#######################################

# Sets start and end year of optimization
start_year = 2024
end_year = 2028

calc_start = datetime(start_year, 1, 1, 00, 00)
calc_end = datetime(end_year, 12, 31, 23, 00)

# sets number of time steps. Set 'None'to get full year,
# else set number of time steps in one year to look at
number_of_time_steps = None #8*24 [h]


# Auxiliary years for myopic optimization
aux_years = True  # Set to True for use of representative years. 
                  # Select the steps [years]; for every xth year between start 
                  # and end year an optimization will be done.
                  # Set to False for optimizing each year individually
aux_year_steps = 5  

# Maximum investment capacity (max_capacity_invest in scenario.xlsx) 
max_cap_once = True # True: limited to whole lifetime of the investment
                    # False: new maximum capacity available in every optimization year


#######################################
#----- Set Optimization Objective
#######################################

# Insert or delete "#" in the beginning of th line to choose objective(s)
objective = [
    # 'Costs',
    'EnvCosts',
    # 'climate change - climate change total',
    # 'JRCII',
    # 'Equilibrium',
    # 'resources - dissipated water',
    # 'resources - fossils',
    # 'resources - land use',
    # 'resources - minerals and metals',
    # 'ecosystem quality - freshwater and terrestrial acidification',
    # 'ecosystem quality - freshwater ecotoxicity',
    # 'ecosystem quality - freshwater eutrophication',
    # 'ecosystem quality - marine eutrophication',
    # 'ecosystem quality - terrestrial eutrophication',
    # 'human health - carcinogenic effects',
    # 'human health - ionising radiation',
    # 'human health - non-carcinogenic effects',
    # 'human health - ozone layer depletion',
    # 'human health - photochemical ozone creation',
    # 'human health - respiratory effects, inorganics'
]

# Multiprocessing for concurrent objectives
# Set to True for parallel execution of objectives, enhancing calculation speed
# Set to False for testing and debugging; debug messages are not displayed during multiprocessing
multiprocessing = False

#######################################
# ------Scenario file
#######################################

# filename of the corresponding spreadsheet with scenario configurations
filename_configuration = 'scenario.xlsx'


#######################################
#----- TMY
#######################################

# download typical meteorological year (TMY) from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY for specific location
# save to folder "in/pvgis" in this directory and change filename below
filename_tmy = 'in/pvgis/tmy_48.642_9.958_2007_2016.csv'


# #######################################
# ----- Location settings
# #######################################
timezone = 'Europe/Berlin'
latitude = 48.642
longitude = 9.958
location_name = 'Test'

#######################################
# ----- Fixed demand
#######################################
# Set to True to get new profiles and save to copy of filename_configuration, sheet "timeseries"
# (csv files in subfolder in/'location_name')
#######################################
# Electricity
#######################################
update_electricity_demand = True

# Sector according to BDEW
ann_el_demand_per_sector = {
    "h0": 400000, # Haushalt
    "g0": 100000, # Gewerbe allgemein             	Gewogener Mittelwert der Profile G1-G6
    # "g1": , # Gewerbe werktags 8–18 Uhr     	    z.B. Büros, Arztpraxen, Werkstätten, Verwaltungseinrichtungen
    # "g2": , # Gewerbe mit starkem bis überwiegendem Verbrauch in den Abendstunden z.B. Sportvereine, Fitnessstudios, Abendgaststätten
    # "g3": , # Gewerbe durchlaufend 	            z.B. Kühlhäuser, Pumpen, Kläranlagen
    # "g4": , # Laden/Friseur 	 
    # "g5": , # Bäckerei mit Backstube 	 
    # "g6": , # Wochenendbetrieb 	                z.B. Kinos
    # "g7": , # Mobilfunksendestation 	            durchgängiges Bandlastprofil
    # "l0": , # Landwirtschaftsbetriebe allgemein 	Gewogener Mittelwert der Profile L1 und L2
    # "l1": , # Landwirtschaftsbetriebe mit Milchwirtschaft/Nebenerwerbs-Tierzucht 	 
    # "l2": , # Übrige Landwirtschaftsbetriebe 	 
    # "h0_dyn": 0, # Haushalt dynamisiert
}
filename_el_demand = 'el_demand_bdew.csv' 
varname_el_demand = 'load_el'

##################################################
# Heat
#################################################
update_heat_demand = True
# heat demand input data for BDEW profile: 
# annual demand per single family house/Einfamilienhaus (efh), 
# multiple family house/Mehrfamilienhaus (mfh), Trade, commerce, services/Gewerbe, Handel, Dienstleistung (ghd)
ann_demands_per_type = {'efh': 0,
                        'mfh': 1000000,
                        'ghd': 20000} # [kWh]

building_class = 10
# class of building according to bdew classification possible numbers are: 1 - 11
# according to https://www.eko-netz.de/files/eko-netz/download/3.5_standardlastprofile_bgw_information_lastprofile.pdf
#    Altbauanteil    mittl. Anteile von
#     von    bis     Altbau   Neubau
# 1  85,5%    90,5%    88,0%    12,0%
# 2  80,5%    85,5%    83,0%    17,0%
# 3  75,5%    80,5%    78,0%    22,0%
# 4  70,5%    75,5%    73,0%    27,0%
# 5  65,5%    70,5%    68,0%    32,0%
# 6  60,5%    65,5%    63,0%    37,0%
# 7  55,5%    60,5%    58,0%    42,0%
# 8  50,5%    55,5%    53,0%    47,0%
# 9  45,5%    50,5%    48,0%    52,0%
# 10 40,5%    45,5%    43,0%    57,0%
# 11                   75,0%    25,0% Durchschnitt DE
# Altbau bis 1979; auf Basis von Wohneinheiten

building_wind_class = 0  # 0=not windy, 1=windy
filename_th_demand = 'heat_demand_bdew.csv' #file will be generated if update_heat_demand = True

separate_heat_water = True  # if True two load curves for space heat and hot water are generated per type
                            # only relevant when update_heat_demand = True

varname_th_low = 'load_th_low' # name used in scenario.xlsx for thermal load for low temperature space heat;
varname_th_high = 'load_th_high' # name used in scenario.xlsx for thermal load for hot water


#######################################
# ----- Renewable Energy Curves
#######################################

# Photovoltaics
#######################################

# optimal/south
update_pv_opt_fix = True
# PV performance as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#PVP 
# for specific location, PV technology, system loss, mounting position, slope, azimuth
# or optimal direction, respectively; 2007-2016
# cryst. Si, 14 % System loss
filename_pv_opt_fix = 'in/pvgis/Timeseries_48.642_9.958_CM_1kWp_crystSi_14_35deg_-3deg_2007_2016.csv'
varname_pv = 'PV_roof_S'
varname_pv_1 = 'PV_flat_roof'
varname_pv_2 = 'PV_og'

# tilt: 180°
update_pv_facade_fix = False
filename_pv_facade_fix = 'in/pvgis/Timeseries_....csv'
varname_pv_3 = 'PV_facade'

# azimuth: 90°
update_pv_west_fix = True
filename_pv_west_fix = 'in/pvgis/Timeseries_48.642_9.958_CM_1kWp_crystSi_14_38deg_90deg_2007_2016.csv'
varname_pv_4 = 'PV_roof_W'

# azimuth: -90°
update_pv_east_fix = True
filename_pv_east_fix = 'in/pvgis/Timeseries_48.642_9.958_CM_1kWp_crystSi_14_38deg_-90deg_2007_2016.csv'
varname_pv_5 = 'PV_roof_E'

####################################################
# Wind
####################################################
update_Wind_data = True
varname_wind = 'wind'

windPlantCapacity = 3200  # [kW]
wind_hub_hight = 142 # [m]
# power curve data from https://www.wind-turbine-models.com
p_in_power_curve = [0.0, 64, 350, 2712, 3200, 3200] # in W for wind speed of 0, 3, 5, 10, 15, 25 m/s
# surface roughness:
# bare rocks, sparely vegetated areas = 0.01 m;
# arable land, pastures, nat. grasslands = 0.1 m; 
# forest = 0.9 m
# [https://www.researchgate.net/figure/Surface-roughness-length-m-for-each-land-use-category-in-WAsP-and-WRF-and-the_tbl1_340415172] 
wind_z0_roughness = 0.1 # length in m; 

##################################################
# Solar Collectors, flat plate
##################################################
update_Solar_Collector_data = True
varname_solar_collector = 'solar_thermal_FPC'
temp_collector_inlet = 30  # [°C] Collectors inlet temperature 
delta_temp_n = 20  # Temperature difference between collector inlet and mean temperature (of inlet and outlet temperature)
collector_tilt = 38
collector_azimuth = 4
a_1 = 3.594  # [W/(m²K)] Thermal loss parameter k1 (https://shop.ssp-products.at/Flachkollektor-SSP-Prosol-25-251m-)
a_2 = 0.014  # [W/(m²K²)] Thermal loss parameter k2
eta_0 = 0.785  # Optical efficiency of the collector


##################################################
# Air/water heatpump
##################################################
update_heatpump_a_w_high_cop = True
varname_a_w_hp_high = 'heat_pump_a_w_high'
hp_temp_high = 55  # maximum outlet temperature in [°C]

update_heatpump_a_w_low_cop = True
varname_a_w_hp_low = 'heat_pump_a_w_low'
hp_temp_low = 40  # maximum outlet temperature in [°C]

a_w_hp_quality_grade = 0.4  # 0.4 is default setting for air/water heat pumps
hp_temp_threshold_icing = 2 # [°C] temperature below which icing occurs at heat exchanger
hp_factor_icing = 0.8 # [0<f<1] sets the relative COP drop by icing; 1 = no efficiency drop


#######################################
#-------Area constraint
#######################################

# sets an constraint to all technologies in sheet "renewables" in scenario file, where row "area" > 0 
area_constraint = True
area = 5500 #[m²]; overall area for restricted technologies


#######################################
# ----- LCA Update
#######################################

# LCA update - get new Excel files from openLCA for active technologies in scenario.xlsx
# if True, will update all LCA data through openLCA (time intensive!), 
# otherwise will only get the missing files from openLCA
update_LCA_data = False
# 'ILCD 2.0 2018 midpoint' #'EF2.0 midpoint' #'ILCD 2.0 2018 midpoint' depending on database in openLCA
LCA_impact_method = 'EF2.0 midpoint'


###############################################################################
# ----- Normalisation & Weighting for objectives JRCII, EnvCosts, Equilibrium
###############################################################################

filename_weight_and_normalisation = 'in/Normalisation and Weighting.xlsx'
# GDP 2010 with current prices for 2023 to be in line with environmental normalisation
normalization_cost_gdp = 6.161E+13

normalisation_per_person = False
normalization_person_population = 6895889018 # population world 2010
# Soruce: United Nations, Department of Economic and Social Affairs, Population Division (2011).
# World Population Prospects: The 2010 Revision, DVD Edition – Extended Dataset (United Nations publication, Sales No. E.11.XIII.7)

# Weighting factor for costs, if scenario requires setting
# For objective EnvCosts weight of Environmental Footprint and costs can be choosen,
# Must be a decimal between 0 and 1 for multi-objective "EnvCosts"
weight_cost_to_env = 0.5 # default: 0.5

# Equilibrium means equal weighting of all impachts but allows for setting all
# weights individually (setting for environmental impacts in Excel file)
weight_cost_to_env_equilibrium = 1/17 # 1/17 if equal weighting between every single goal for multi-objective "Equilibrium"


#######################################
# ----- Emission constraint
#######################################

emission_constraint = False
ec_horizon = 5
ec_impact_category = 'climate change - climate change total'
ec_buffer = 0.0001
ef_fuel_based_only = False


#######################################
# ----- Climate neutrality
#######################################

def_cn_calculate_climate_neutrality = False
def_cn_include_investment = True
def_cn_year_climate_neutrality = 2045
def_cn_fuel_based_only = False


#######################################
# ----- Financial settings
#######################################

# weighted average cost of capital (wacc), part of annuity calculation based on oemof.economics.annuity
InvestWacc = 0  
# kW (kWh for storage) investments below this value do not get passed as existing capacity to next year
Invest_min_threshhold = 0.1


###############################################################################
# Technical settings
###############################################################################

# Set the logging level
log_screen_level = logging.INFO
log_file_level = logging.DEBUG

# Results for invest capacities and flows per year are displayed in a table per
# optimization goal and year on the console if True
showTable = True

# changing the following config variables may result in errors! Proceed with caution!

# List of possible investment classes
ci = ['transformers_in', 'renewables', 'storages', 'transformers_out']

InvestTimeSteps = 1 if aux_years == False else aux_year_steps  # years

# granularity of calculation; chose 'D' for daily or 'H' for hourly
granularity = 'H'

# Solver
solver = 'cbc'  # 'cplex', 'glpk', 'gurobi',....
solver_verbose = True  # show/hide solver output
solver_options_on = False
solver_options = {
    'threads': 4,
    'method': 2,  # barrier
    'crossover': 0,
    'BarConvTol': 1.e-5,
    'FeasibilityTol': 1.e-6,
    'AggFill': 0,
    'PreDual': 0,
    'GURO_PAR_BARDENSETHRESH': 200,
    'nodefilestart': 0.5
} if solver_options_on == True else {}

# Do NOT change this list unless you're changing the impact assessment methed with 
# associated weighting & normalization
system_impacts_index = [
    'Costs',
    'climate change - climate change biogenic',
    'climate change - climate change fossil',
    'climate change - climate change land use and land use change',
    'climate change - climate change total',
    'ecosystem quality - freshwater and terrestrial acidification',
    'ecosystem quality - freshwater ecotoxicity',
    'ecosystem quality - freshwater eutrophication',
    'ecosystem quality - marine eutrophication',
    'ecosystem quality - terrestrial eutrophication',
    'human health - carcinogenic effects',
    'human health - ionising radiation',
    'human health - non-carcinogenic effects',
    'human health - ozone layer depletion',
    'human health - photochemical ozone creation',
    'human health - respiratory effects, inorganics',
    'resources - dissipated water',
    'resources - fossils',
    'resources - land use',
    'resources - minerals and metals',
    'JRCII',
    'EnvCosts',
    'Equilibrium'
]
system_impacts = pd.DataFrame(index=system_impacts_index)


def createLogPathName():
    now = datetime.now()
    time = str(now)
    time = time[:-7]
    time = time.replace(':', '-')

    # creates name for folder where all data is stored
    name = os.path.dirname(os.path.realpath(__file__)) + '\\' + 'runs' + '\\' + str(now.date()) + '_' + str(
        now.hour) + '-' + str(now.minute) + '-' + str(now.second)

    # if folder does not exist, creates folder and copies config.py and scenario.xlsx
    if not os.path.exists(name):
        os.makedirs(name)

        # copies config.py to runs folder
        src = os.path.dirname(os.path.realpath(__file__)) + '\\' + 'config.py'
        dst = name + '\\' + f'laend_config_{time}.py'
        shutil.copyfile(src, dst)

        # creates subfolder for files
        if not os.path.exists(name + '\\' + 'files'):
            os.makedirs(name + '\\' + 'files')

        # creates subfolder for timeseries
        if not os.path.exists('in' + '\\' + location_name):
            os.makedirs('in' + '\\' + location_name)

            # creates subfolder for oemof dumps
        if not os.path.exists(name + '\\' + 'oemof_dumps'):
            os.makedirs(name + '\\' + 'oemof_dumps')

        # copies scenario.xlsx to files folder
        src = os.path.dirname(os.path.realpath(__file__)) + \
            '\\' + filename_configuration
        dst = name + '\\' 'files\\' + f'{time}_{filename_configuration}'
        shutil.copyfile(src, dst)

    if not os.path.exists(name + '\\logs'):
        os.mkdir(name + '\\logs')

    return name, time
