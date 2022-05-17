import pandas as pd
import os
import shutil
from datetime import datetime

import logging

#showPlot = False

#######################################
#------Model specific settings
#######################################
#----- Temporal Settings
#######################################

# Start and end year of optimization
start_year = 2022
end_year = 2026

calc_start = datetime(start_year, 1, 1, 00, 00)
calc_end = datetime(end_year, 12, 31, 23, 00)

# sets number of time steps. set 'None'to get full year, else set number of time steps in one year to look at
number_of_time_steps = 8*24


# Auxiliary years (5 year steps)f
aux_years = False # True to use representative years and select the steps. False to optimize each year individually
aux_year_steps = 5 #years


#######################################
#----- Set Optimization Objective
#######################################

objective = [
    'Costs',
    'EnvCosts',
    # 'climate change - climate change total',
    # 'JRCII',
    # 'Equilibrium',
    # 'resources - dissipated water',
    # 'resources - fossils',
    # 'resources - land use',
    # 'resources - minerals and metals',
    # 'climate change - climate change biogenic',
    # 'climate change - climate change fossil',
    # 'climate change - climate change land use and land use change',
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
# Multiprocessing of several objectives
# True if several objectives should run in parallel to speed up the calculation time
# False for testing and to use debug mode, since debug messages are not displayed
# during multiprocessing
multiprocessing = False

#######################################
#------Scenario Excel file
#######################################

filename_configuration = 'scenario.xlsx'

#######################################
#------Location specific settings
#######################################
#----- TMY
#######################################

# import typical meteorological year (TMY) as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY for specific location
filename_tmy = 'in/pvgis/tmy_47.658_9.169_2007_2016.csv'


# #######################################
# ----- Location settings
# #######################################
timezone='Europe/Berlin',
latitude=47.658
longitude=9.169


#######################################
#----- Fixed demand
#######################################
# Get new electricity profile and save to filename_configuration (copy of Scenario Excel in files folder)
update_electricity_demand = False
filename_el_demand = 'in/el_demand_bdew.csv'
varname_el_demand = 'load_el'

# Get new heat profile and save to filename_configuration (copy of Scenario Excel in files folder)
update_heat_demand = False
filename_th_demand = 'in/heat_demand_bdew.csv'
varname_th_low = 'load_th_low' #name used in scenario.xlsx for thermal load for low temperature space heat; 
varname_th_high = 'load_th_high' #name used in scenario.xlsx for thermal load for hot water


separate_heat_water = True

#BDEW heat demand input data
ann_demands_per_type = {'efh': 0,
                        'mfh': 960990,
                        'ghd': 19613}

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

building_wind_class = 0 # 0=not windy, 1=windy


#######################################
#----- Renewable Energy Curves
#######################################

update_pvgis_data = False
#PV performance as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#PVP for specific location, PV technology, system loss, mounting position, slope, azimuth
filename_pvgis = 'in/pvgis/Timeseries_47.658_9.169_SA_1kWp_crystSi_14_36deg_4deg_2007_2016.csv' #crist. Si, 14 % Systemverlust
varname_pv_1 = 'PV_roof'
varname_pv_2 = 'PV_og'


update_Wind_data = False
varname_wind = 'wind'

windHeightOfData = {
    'dhi': 0,
    'dirhi': 0,
    'pressure': 244,
    'temp_air': 377,
    'v_wind': 377,
    'Z0': 0}


windPowerPlant = {
    'h_hub': 70,
    'd_rotor': 50,
    'wind_conv_type': 'NORDEX N50 800'}


windPlantCapacity = 800 #kW
wind_z0_roughness = 0.1


# Solar Collector data (https://shop.ssp-products.at/Flachkollektor-SSP-Prosol-25-251m-)
update_Solar_Collector_data = False
filename_solar_collector_high = 'solar_thermal_FPC_high'
filename_solar_collector_low = 'solar_thermal_FPC_low'
collector_tilt = 36
collector_azimuth = 5
a_1 = 3.594 #[W/(m²K)] Thermal loss parameter k1
a_2 = 0.014 #[W/(m²K²)] Thermal loss parameter k2
eta_0 = 0.785 #Optical efficiency of the collector
temp_collector_inlet = 50 #[°C] Collectors inlet temperature
delta_temp_n = 10 #Temperature difference between collector inlet and mean temperature


# heatpump air water data
update_heatpump_a_w_cop = False

varname_a_w_hp_high = 'heat_pump_a_w_high'
hp_temp_high = 55 #°C

varname_a_w_hp_low = 'heat_pump_a_w_low'
hp_temp_low = 40 #°C

a_w_hp_quality_grade = 0.4 #0.4 is default setting for air/water heat pumps 
hp_temp_threshold_icing = 2 #[°C] temperature below which icing occurs at heat exchanger
hp_factor_icing = 0.8 #[0<f<1] sets the relative COP drop by icing; 1 = no efficiency drop


#######################################
#----- LCA Update
#######################################

# LCA update - get new Excel files from openLCA for active technologies in scenario.xlsx 
update_LCA_data = False # if True, will update all LCA data through openLCA (time intensive!), otherwise will only get the missing files from openLCA
LCA_impact_method = 'EF2.0 midpoint' #'ILCD 2.0 2018 midpoint' #'EF2.0 midpoint' #'ILCD 2.0 2018 midpoint' depending on database in openLCA


# Do NOT change this list unless you're changing the impact assessment methed with associated weighting & normalization
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
system_impacts = pd.DataFrame(index= system_impacts_index)

###############################################################################
#----- Normalisation & Weighting for objectives JRCII, EnvCosts, Equilibrium
###############################################################################

filename_weight_and_normalisation = 'in/Normalisation and Weighting.xlsx'
normalization_cost_gdp = 4.63113E+13 #GDP 2018 with current prices for 2010 to be in line with environmental normalisation (for calculation see GDP_2010_Euro.xlsx) (todo: auf 2021 aktualisieren)

normalisation_per_person = False #True
normalization_person_population = 6895889018 #United Nations, Department of Economic and Social Affairs, Population Division (2011). World Population Prospects: The 2010 Revision, DVD Edition – Extended Dataset (United Nations publication, Sales No. E.11.XIII.7)

# Weighting factor for costs, if scenario requires setting
# For objective EnvCosts weight of Environmental Footprint and costs can be choosen, 
weight_cost_to_env = 0.5 #Must be a decimal between 0 and 1 for multi-objective "EnvCosts"
# Equilibrium means equal weighting of all impachts but allows for setting all weights individually (setting for environmental impacts in Excel file)
weight_cost_to_env_equilibrium = 1/17 #1/17 if equal weighting between every single goal for multi-objective "Equilibrium"


#######################################
#----- Emission constraint
#######################################

emission_constraint = False
ec_horizon = 5
ec_impact_category = 'climate change - climate change total'
ec_buffer = 0.0001
ef_fuel_based_only = False


#######################################
#----- Climate neutrality
#######################################

def_cn_calculate_climate_neutrality = False
def_cn_include_investment = True
def_cn_year_climate_neutrality = 2045
def_cn_fuel_based_only = False


#######################################
#----- Financial settings
#######################################

#InvestCostDecrease = 0.01  # Annual cost decrease (technical progress)

InvestWacc = 0.01  # weighted average cost of capital, part of annuity calculation based on oemof.economics.annuity
Invest_min_threshhold = 0.1 #kW (kWh for storage) investments below this value do not get passed as existing capacity to next year
InvestTimeSteps = 1 if aux_years == False else aux_year_steps #years
Invest_sell_if_unused = False


###############################################################################
# Technical settings
###############################################################################

# Set the logging level
log_screen_level = logging.INFO
log_file_level = logging.DEBUG


# changing the following config variables may result in errors! Proceed with caution!
ci = ['transformers_in', 'renewables', 'storages', 'transformers_out']

# granularity of calculation; chose 'D' for daily or 'H' for hourly
granularity = 'H'

# Solver
solver = 'cbc' #'cplex', 'glpk', 'gurobi',....
solver_verbose = True  # show/hide solver output
solver_options_on = False
solver_options = {
    'threads': 4,
    'method': 2, # barrier
    'crossover': 0,
    'BarConvTol': 1.e-5,
    'FeasibilityTol': 1.e-6,
    'AggFill': 0,
    'PreDual': 0,
    'GURO_PAR_BARDENSETHRESH': 200,
    'nodefilestart':0.5
} if solver_options_on == True else {}

continue_despite_storage_issues = None

def createLogPathName():
    now = datetime.now()
    time = str(now)
    time = time[:-7]
    time = time.replace(':','-')
   
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

            # creates subfolder for files
        if not os.path.exists(name + '\\' + 'oemof_dumps'):
            os.makedirs(name + '\\' + 'oemof_dumps')

        # copies scenario.xlsx to files folder
        src = os.path.dirname(os.path.realpath(__file__)) + '\\' + filename_configuration
        dst = name + '\\' 'files\\' + f'{time}_{filename_configuration}'
        shutil.copyfile(src, dst)

    if not os.path.exists(name + '\\logs'):
        os.mkdir(name + '\\logs')

    return name, time
