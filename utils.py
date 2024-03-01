# import general libraries
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import calendar
from matplotlib import pyplot as plt
import os
import sys

# import oemof
import oemof.solph as solph
from oemof.tools import economics

# import openLCA connection
import olca

# import special oemof libraries for demand and supply data
from windpowerlib import ModelChain, WindTurbine, create_power_curve
from oemof.thermal import solar_thermal_collector
import oemof.thermal.compression_heatpumps_and_chillers as cmpr_hp_chiller
import demandlib.bdew as bdew
import demandlib.particular_profiles as profiles

# import files
import config

import warnings
warnings.simplefilter("ignore")

################################
#----Functions used in main()
################################

def defineYearsForCalculation():
    '''
    Creates an array of years used in the calculation. 
    Based on config.aux_years (t/f) and config.aux_year_steps
    Used in writeParametersLogger
    :return: calc_years e.g. [2020,2025]
    '''
    calc_years = []
    start = config.start_year

    if config.aux_years == True:
        for i in range(config.start_year + 1, config.end_year + 1):
            while start < config.end_year + 1:
                calc_years.append(start)
                start += config.aux_year_steps

    else:
        for i in range(config.start_year, config.end_year + 1):
            while start < config.end_year + 1:
                calc_years.append(start)
                start += 1

    return calc_years


def writeParametersLogger():
    '''
    Writes input Parameters to logger for start of calculation.
    :return: start_time, info dictionary for excel export
    '''
    info = {}
    now = datetime.now()
    logging.info('Started LAEND II at: ' + str(now))
    info['Calculation start time'] = now
    logging.info('***********Parameters used in calculation:*********')
    logging.info('Start year: ' + str(config.start_year))
    info['Start year'] = config.start_year
    logging.info('End year: ' + str(config.end_year))
    info['End year'] = config.end_year
    logging.info('Granularity: ' + str(config.granularity))
    info['Granularity'] = config.granularity
    logging.info(f'Years selected for calculation: {defineYearsForCalculation()}')
    info['Years for calculation'] = defineYearsForCalculation()
    info[
        'Investment time steps (annual result will be multiplied with this number \
        if aux years are used!) '] = config.InvestTimeSteps
    info['Emission Constraint'] = config.emission_constraint
    logging.info('Using emission constraints: ' + str(config.emission_constraint))

    for i in config.objective:
        logging.info(f'Objectives: {i}')
    logging.info('***********++++++++++++++++++++++++++++++*********')



def compileTMY(file_name):
    '''
    Takes pvgis tmy file and prepares it for later use; 
    change of timeindex to 2016 and insertion of leap day
    :param file_name: csv file of tmy data from pvgis
    :return: tmy file for 2016 with leap day, series of month and year that was chosen for tmy
    '''

    # create dictionary of years chosen for particular month
    my = pd.read_csv(file_name, skiprows=range(0, 3), nrows=12, sep=',')
    my.index = my.month
    del my['month']
    tmy_month_year = pd.Series(my['year']).to_dict()

    # get hourly data
    df = pd.read_csv(file_name, sep=',', skiprows=range(0, 16))
    df = df.drop(df.index[8760:])
    del df['time(UTC)']
    df = df.apply(pd.to_numeric)

    #### create leap day

    # set index do non leap year
    df.index = pd.date_range(
        start=datetime(2015, 1, 1, 00, 00), end=datetime(2015, 12, 31, 23, 00), freq='H')

    # change the year to leap year = data without february 29
    df.index = df.index.map(lambda t: t.replace(year=2016))

    # get data from Feb 23 to March 5 and turn it into an average day
    average_day = df.loc['2016-02-23': '2016-03-05']
    average_day = average_day.groupby(by=average_day.index.hour).mean()
    average_day.index = pd.date_range(start=datetime(2016, 2, 29, 00, 00), \
                                      end=datetime(2016, 2, 29, 23, 00), freq='H')

    # add the average day
    tmy = pd.concat([df, average_day], sort=True).sort_index()

    return tmy, tmy_month_year


def getElectricityDemand(ann_el_demand_per_sector, run_name, time):
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

   
    # Resample 15-minute values to hourly values.
    elec_demand_resampled = elec_demand.resample("H").mean()
    demand = elec_demand_resampled

    demand.to_csv(f'in/{config.location_name}/{config.varname_el_demand}')
    
    importFixedFlow(
        run_name, time, f'in/{config.location_name}/{config.varname_el_demand}', 
        'demand', config.varname_el_demand, sum_mult_profiles=True
        )
    
def getHeatDemand(testmode=False, ann_demands_per_type=None, temperature=None):
    '''
    Used bdew demandlib to create heat demand curve
    https://demandlib.readthedocs.io/en/latest/

    :param testmode:
    :param ann_demands_per_type:
    :param temperature:
    :return:
    '''

    if ann_demands_per_type is None:
        ann_demands_per_type = {'efh': 1/3,
                                'mfh': 1/3,
                                'ghd': 1/3}
    holidays = {
        datetime(2016, 5, 16): 'Whit Monday',
        datetime(2016, 3, 28): 'Easter Monday',
        datetime(2016, 5, 5): 'Ascension Thursday',
        datetime(2016, 1, 1): 'New year',
        datetime(2016, 10, 3): 'Day of German Unity',
        datetime(2016, 12, 25): 'Christmas Day',
        datetime(2016, 5, 1): 'Labour Day',
        datetime(2016, 3, 25): 'Good Friday',
        datetime(2016, 12, 26): 'Second Christmas Day'}

    # Create DataFrame for 2016
    demand = pd.DataFrame(
        index=pd.date_range(datetime(2016, 1, 1, 0),
                            periods=8784, freq='H'))

    # Single family house (efh: Einfamilienhaus)
    if "efh" in ann_demands_per_type:
        demand['efh'] = bdew.HeatBuilding(
            demand.index, holidays=holidays, temperature=temperature,
            shlp_type='EFH',
            building_class=config.building_class, wind_class=config.building_wind_class,
            annual_heat_demand=ann_demands_per_type['efh'],
            name='EFH').get_bdew_profile()

    # Multi family house (mfh: Mehrfamilienhaus)
    if "mfh" in ann_demands_per_type:
        demand['mfh'] = bdew.HeatBuilding(
            demand.index, holidays=holidays, temperature=temperature,
            shlp_type='MFH',
            building_class=config.building_class, wind_class=config.building_wind_class,
            annual_heat_demand=ann_demands_per_type['mfh'],
            name='MFH').get_bdew_profile()

    # Industry, trade, service (ghd: Gewerbe, Handel, Dienstleistung)
    if "ghd" in ann_demands_per_type:
        demand['ghd'] = bdew.HeatBuilding(
            demand.index, holidays=holidays, temperature=temperature,
            shlp_type='ghd', wind_class=config.building_wind_class,
            annual_heat_demand=ann_demands_per_type['ghd'],
            name='ghd').get_bdew_profile()

    # functionality from bdew example, not tested
    if not testmode:
        if plt is not None:
            # Plot demand of building
            ax = demand.plot()
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat demand in kW")
            plt.show()
        else:
            logging.info('Annual consumption: \n{}'.format(demand.sum()))

    if config.separate_heat_water == True:

        for i in list(demand.columns):
            
            mn = demand[i].loc['2016-07-01':'2016-08-31'].mean()
            a = np.array(demand[i].values.tolist())

            demand[f'{i}_water'] = np.where(a > mn, mn, a).tolist()
            demand[f'{i}_heat'] = demand[i] - demand[f'{i}_water']
        
        demand['total_water'] = demand['efh_water'] + demand['mfh_water'] + demand['ghd_water']
        demand['total_heat'] = demand['efh_heat'] + demand['mfh_heat'] + demand['ghd_heat']
        
    else:
        demand['total'] = demand['efh'] + demand['mfh'] + demand['ghd']
                    
    demand.to_csv(f'in/{config.location_name}/{config.filename_th_demand}')
    return demand


def importFixedFlow(run_name, time, file_name, sheet_of_item, item_name, 
                    sum_mult_profiles=False, col_name=None, conversion=None):
    '''
    Takes Excel table and saves fixed flow to copy of scenario file in files folder
    Imported data should have an hourly granularity and start on January 1 at 0:00!
    #ToDo Takes and saves as excel, not great performance. Speed can be improved here.

    :param file_name: File name of file where data for fixed flow is saved
    :param sheet_of_item: Sheet name in scenario file where item e.g. demand or pv in renewables is saved
    :param item_name: Item name = label in scenario file 
    :param sum_mult_profiles: multiple profiles are summed into one fixed flow. Will sum all flows!
    :param col_name: Single column to be used as fixed flow
    :param conversion: Should profile be converted e.g. 1/1000 for conversion W to kW
    :return: saves as xlsx file
    '''
    logging.info(f'Importing {file_name}')
    xls = pd.ExcelFile(f'{run_name}\\files\\{time}_{config.filename_configuration}')

    data = {}
    for sheet in xls.sheet_names:
        data[sheet] = xls.parse(sheet)

    if not item_name in data[sheet_of_item]['label'].tolist():
        raise ValueError(
            f'Please ensure that the bus "{item_name}" is included in the file \
            {config.filename_configuration}')

    profiles = pd.read_csv(file_name)

    if len(data['timeseries']) > len(profiles):
        raise ValueError(
            'Length of imported profile is not long enough to account for 8784 hours in a leap year.')

    elif len(data['timeseries']) < len(profiles):
        logging.warning(
            f'Length of imported profile is not equal to number of hours in leap \
            year (8784) but {len(profiles)}. Import will cut off excessive rows at the end.')
        profiles = profiles[:len(data['timeseries'])]

    # creates sum of one row to sum multiple profiles
    if sum_mult_profiles == True:
        profiles = profiles.sum(axis=1)

    # takes only one column e.g. energy output of pv in pvgis weather data
    if col_name != None:
        profiles = profiles[col_name]

    # multiplies with conversion, e.g. MJ to kWh
    if conversion != None:
        profiles = profiles * conversion

    data['timeseries'][f'{item_name}.fix'] = profiles

    pos = data[sheet_of_item]['label'].tolist().index(item_name)
    data[sheet_of_item].at[pos, 'fixed'] = 1

    # writes profile to excel table for later input
    writer = pd.ExcelWriter(
        f'{run_name}\\files\\{time}_{config.filename_configuration}', engine='xlsxwriter')

    for key in data.keys():
        data[key].to_excel(writer, sheet_name=key, index=False)

    writer.save()

    logging.info(f'Successfully imported {file_name} as fixed flow for {item_name}')


def importFixedCOP(run_name, time, file_name, sheet_of_item, item_name, 
                   sum_mult_profiles=False, 
                   col_name=None,
                   conversion=None):
    '''
    Takes Excel table and saves fixed flow to copy of scenario file
    Imported data should have an hourly granularity and start on January 1 at 0:00!
    
    :param file_name: File name of file where data for fixed flow is saved
    :param sheet_of_item: Sheet name in scenario file where item e.g. demand or pv in renewables is saved
    :param item_name: Item name = label in scenario file 
    :param sum_mult_profiles: multiple profiles are summed into one fixed flow. Will sum all flows!
    :param col_name: Single column to be used as fixed flow
    :param conversion: Should profile be converted e.g. 1/1000 for conversion W to kW
    :return: saves to Excel
    '''
    logging.info(f'Importing {file_name}')
    xls = pd.ExcelFile(f'{run_name}\\files\\{time}_{config.filename_configuration}')

    data = {}
    for sheet in xls.sheet_names:
        data[sheet] = xls.parse(sheet)

    if not item_name in data[sheet_of_item]['label'].tolist():
        raise ValueError(
            f'Please ensure that the bus "{item_name}" is included in the file \
            {config.filename_configuration}')

    profiles = pd.read_csv(file_name)

    if len(data['timeseries']) > len(profiles):
        raise ValueError('Length of imported profile is not long enough to account \
                         for 8784 hours in a leap year.')

    elif len(data['timeseries']) < len(profiles):
        logging.warning(
            f'Length of imported profile is not equal to number of hours in leap \
            year (8784) but {len(profiles)}. Import will cut off excessive rows at the end.')
        profiles = profiles[:len(data['timeseries'])]

    # creates sum of one row to sum multiple profiles
    if sum_mult_profiles == True:
        profiles = profiles.sum(axis=1)

    # takes only one column e.g. energy output of pv in pvgis weather data
    if col_name != None:
        profiles = profiles[col_name]

    # multiplies with conversion, e.g. MJ to kWh
    if conversion != None:
        profiles = profiles * conversion

    data['timeseries'][f'{item_name}.cop'] = profiles

    pos = data[sheet_of_item]['label'].tolist().index(item_name)
    data[sheet_of_item].at[pos, 'fixed_cop'] = 1

    # writes profile to excel table for later input
    writer = pd.ExcelWriter(
        f'{run_name}\\files\\{time}_{config.filename_configuration}', engine='xlsxwriter')

    for key in data.keys():
        data[key].to_excel(writer, sheet_name=key, index=False)

    writer.save()

    logging.info(f'Successfully imported {file_name} as fixed cop for {item_name}')


def createPvProfileForTMY(file_name, tmy_month_year, name):
    logging.info('Creating pv profile based on typical meteorological year (tmy)')

    # get df from pvgis csv export (must include all years in tmy)
    pv = pd.read_csv(file_name, skiprows=range(0, 10), sep=',')
    pv = pv.drop(pv.index[len(pv) - 7:])
    pv['time'] = pd.to_datetime(pv['time'], format='%Y%m%d:%H%M')
    pv['time'] = [i.replace(minute=0) for i in pv['time']]
    pv.index = pv['time']

    # pull the relevant month and create one typical year
    df = pd.DataFrame()

    for month in tmy_month_year.keys():
        year = tmy_month_year[month]

        data = pv.loc[pv.index.year == year]
        data = data.loc[data.index.month == month]

        df = pd.concat([df, data], sort=True)

    df.index = df.index.map(lambda t: t.replace(year=2016))  
    
    
    # get data from Feb 23 to March 5 and turn it into an average leap day
    # to be included: check if leap day exists; if Feb in TMY is leap year 29.2. already exists
    # and will be double;  
  
    if len(df) < 8784:
    # if pd.timestamp('2016-02-29 00:00:00') in df_index == False:
        
        average_day = df.loc['2016-02-23': '2016-03-05']
        average_day = average_day.apply(pd.to_numeric)
    
        leap_day = average_day.groupby(by=average_day.index.hour).mean()
        leap_day.index = pd.date_range(
            start=datetime(2016, 2, 29, 00, 00), end=datetime(2016, 2, 29, 23, 00), freq='H'
            )
        leap_day['time'] = leap_day.index
    
        # add the average leap day to data
        df = pd.concat([df, leap_day], sort=True).sort_index()

    # export
    df.to_csv(f'in/{config.location_name}/pvgis_tmy_{name}.csv')

    logging.info('pv profile adaption successful')


def createWindPowerPlantFixedFlow(item_name, my_weather, run_name, time):
    '''
    Windpowerplant settings in config file also inputs.

    :param item_name: wind_plant_name
    :param my_weather: tmy compiled with compileTMY()
    :return: exports to excel, does not return anything
    '''

    logging.info(f'Creating fixed flow for wind: {item_name}')
    
    timezone = 'Europe/Berlin'
    wind_z0_roughness = config.wind_z0_roughness

    my_weather = my_weather.copy()
    # convert tmy to required format
    my_weather.index = pd.to_datetime(my_weather.index, utc=True)
    my_weather.index = my_weather.index.tz_convert(
            timezone)
    my_weather = my_weather.rename(columns={'WS10m': 'v_wind', 'SP': 'pressure', 'T2m': 'temp_air'})
    my_weather['temp_air'] += 273.15 # converts °C in K
    my_weather['z0'] = wind_z0_roughness

    # The columns of the DataFrame my_weather are a MultiIndex where the first level
    # contains the variable name as string (e.g. 'wind_speed') and the
    # second level contains the height as integer at which it applies
    # (e.g. 10, if it was measured at a height of 10 m). The index is a
    # DateTimeIndex.
    
    windHeightOfData = { # height in m over ground of mesurement of single parameter
        # 'dhi': 0,
        # 'dirhi': 0,
        'pressure': 0, # height in which pressure was measured
        'temp_air': 2, # height in which temperature was measured
        'v_wind': 10, # height in which wind speed was measured
        'Z0': 0} # height in which roughness was measured
    
    # insert second column name for height of data 
    my_weather = my_weather[["v_wind", "temp_air", "z0", "pressure"]]
    my_weather.columns = [
        ["wind_speed", "temperature", "roughness_length", "pressure"],
        [
         windHeightOfData["v_wind"], 
         windHeightOfData["temp_air"],
         windHeightOfData["Z0"],
         windHeightOfData["pressure"],
        ],
    ]
    
    
    # Specification of wind turbine
    # ************************************************************************
    # **** Data is provided in the oedb turbine library **********************
    
    # windPowerPlant = config.windPowerPlant
    # windPowerPlant = WindTurbine(**windPowerPlant)
    
    # ************************************************************************
    # **** Specification of wind turbine with your own data ******************
    # **** NOTE: power values and nominal power have to be in Watt
    
    windPowerPlant = {
        "nominal_power": config.windPlantCapacity * 1000,  # in W
        "hub_height": config.wind_hub_hight,  # in m
        "power_curve": pd.DataFrame(
            data={
                "value": [
                    p * 1000
                    for p in config.p_in_power_curve
                ],  # in W
                "wind_speed": [0.0, 3.0, 5.0, 10.0, 15.0, 25.0],
            }
        ),  # in m/s
    }
    windPowerPlant = WindTurbine(**windPowerPlant)
    
    # Use of ModelChain
    # ************************************************************************
    # **** ModelChain with default parameter *********************************
    mc_windPowerPlant = ModelChain(windPowerPlant).run_model(my_weather)
    # write power output time series to WindTurbine object
    windPowerPlant.power_output = mc_windPowerPlant.power_output
    
    # ************************************************************************
    # **** ModelChain with non-default value for "wind_speed_model" **********
    # mc_example_turbine = ModelChain(
    #     windPowerPlant, wind_speed_model="hellman").run_model(my_weather)
    # windPowerPlant.power_output = mc_example_turbine.power_output
    
    # ************************************************************************
    # **** ModelChain with non-default specifications ************************
    # modelchain_data = {
    #     "wind_speed_model": "logarithmic",  # 'logarithmic' (default),
    #     # 'hellman' or
    #     # 'interpolation_extrapolation'
    #     "density_model": "ideal_gas",  # 'barometric' (default), 'ideal_gas' or
    #     # 'interpolation_extrapolation'
    #     "temperature_model": "linear_gradient",  # 'linear_gradient' (def.) or
    #     # 'interpolation_extrapolation'
    #     "power_output_model": "power_coefficient_curve",  # 'power_curve'
    #     # (default) or 'power_coefficient_curve'
    #     "density_correction": True,  # False (default) or True
    #     "obstacle_height": 0,  # default: 0
    #     "hellman_exp": None,
    # }  # None (default) or None
    # # initialize ModelChain with own specifications and use run_model method
    # # to calculate power output
    # mc_windPowerPlant = ModelChain(windPowerPlant, **modelchain_data).run_model(my_weather)
    # # write power output time series to WindTurbine object
    # windPowerPlant.power_output = mc_windPowerPlant.power_output
    
    # ************************************************************************
    
    wind_plant_fix = windPowerPlant.power_output/ 1000  # W --> kW
    wind_plant_fix.name = item_name

    wind_plant_fix.to_csv(f'in/{config.location_name}/{item_name}.csv', index=False, header=item_name)

    importFixedFlow(run_name, time, f'in/{config.location_name}/{item_name}.csv', 
                    'renewables', item_name, conversion=1 / config.windPlantCapacity)
        


def createSolarCollectorFixedFlow(item_name, my_weather, run_name, time):
    logging.info(f'Creating fixed flow for solar collector: {item_name}')

    df = my_weather.copy()

    # get date as index
    df['Date'] = df.index

    # localize data for irradiance purposes, the resulting df does not have 
    # daylight savings time rows (spring and fall)
    df.index = df.index.tz_localize(tz='Europe/Berlin', ambiguous='NaT', nonexistent='NaT')
    df = df[pd.notnull(df.index)]

    # create collector_heat dataframe
    precal_data = solar_thermal_collector.flat_plate_precalc(
        config.latitude,
        config.longitude,
        config.collector_tilt,
        config.collector_azimuth,
        config.eta_0,
        config.a_1,
        config.a_2,
        config.temp_collector_inlet,
        config.delta_temp_n,
        irradiance_global=df['G(h)'],
        irradiance_diffuse=df['Gd(h)'],
        temp_amb=df['T2m'],
    )

    # add info to my_weather df and go back to original index
    df['collectors_heat'] = precal_data['collectors_heat']
    df.index = pd.to_datetime(df['Date'])

    # add the new daylight savings rows as 0
    new = pd.DataFrame(
        index=pd.date_range(start=datetime(2016, 1, 1, 00, 00), end=datetime(2016, 12, 31, 23, 00),
                            freq='H'))
    new['collectors_heat'] = df['collectors_heat']
    new = new.fillna(0)

    # export as csv
    new.to_csv(f'in/{config.location_name}/{item_name}.csv', index=False)

    importFixedFlow(run_name, time, f'in/{config.location_name}/{item_name}.csv', \
                    'renewables', item_name, conversion=(1 / 1000))


def createHeatpumpAirFixedCOP(item_name, temp_high, my_weather, run_name, time):
    logging.info(f'Creating fixed COP for heat pump air water: {item_name}')

    # Precalculation of COPs
    cops = cmpr_hp_chiller.calc_cops(
        temp_high=[temp_high],
        temp_low=my_weather['T2m'],
        quality_grade=config.a_w_hp_quality_grade,
        mode='heat_pump',
        temp_threshold_icing=config.hp_temp_threshold_icing,
        factor_icing=config.hp_factor_icing)

    # add the new daylight savings rows as 0
    new = pd.DataFrame(
        index=pd.date_range(start=datetime(2016, 1, 1, 00, 00), end=datetime(2016, 12, 31, 23, 00),
                            freq='H'))
    new[item_name] = cops

    # export as csv
    new.to_csv(f'in/{config.location_name}/{item_name}_cops.csv', index=False)

    importFixedCOP(run_name, time, f'in/{config.location_name}/{item_name}_cops.csv', \
                   'transformers_out', item_name)


def importNodesFromExcel(filename):
    """Read node data from scenario file

    Parameters
    ----------
    filename : :obj:`str`
        Path to excel file

    Returns
    -------
    :obj:`dict`
        Imported nodes data
    """

    # does Excel file exist?
    if not filename or not os.path.isfile(filename):
        raise FileNotFoundError("Excel data file {} not found.".format(filename))
    
    logging.info("Import of Excel data")
    # read excel file
    xls = pd.ExcelFile(filename)

    # parse sheets of excel file
    nodes_data = {
        "buses": xls.parse("buses"),
        "commodity_sources": xls.parse("commodity_sources"),
        "transformers_in": xls.parse("transformers_in"),
        "transformers_out": xls.parse("transformers_out"),
        "renewables": xls.parse("renewables"),
        "demand": xls.parse("demand"),
        "storages": xls.parse("storages"),
        "timeseries": xls.parse("timeseries"),
    }

    # set index in timeseries to original index
    dtindex = pd.to_datetime(nodes_data['timeseries']['timestamp'])
    nodes_data['timeseries'] = nodes_data['timeseries'].set_index(dtindex)

    # resample for daily granularity
    if config.granularity == 'D':
        nodes_data['timeseries'] = nodes_data['timeseries'].resample('D').sum()

    # add columns for month, day and hour
    nodes_data['timeseries']['month'] = nodes_data['timeseries'].index.month
    nodes_data['timeseries']['day'] = nodes_data['timeseries'].index.day

    if config.granularity == 'H':
        nodes_data['timeseries']['hour'] = nodes_data['timeseries'].index.hour

    logging.info("Data from scenario file {} imported.".format(filename))

    sheets = list(nodes_data.keys())
    sheets.remove('timeseries')

    # iterate through sheets
    for i in sheets:

        lst = []
        # iterate through list of possible investments e.g. for transformers
        for num, x in nodes_data[i].iterrows():

            if pd.isna(x['label']):
                lst.append(num)

            # only keeps the active elements
            elif nodes_data[i].iloc[num]['active'] != 1:
                logging.info(nodes_data[i].iloc[num]['label'] + ' is not active')
                lst.append(num)

        nodes_data[i] = nodes_data[i].drop(labels=lst, axis=0)

        un = [x for x in nodes_data[i].columns if str(x).__contains__('Unnamed') or \
                                                  str(x).__contains__('warnings') or \
                                                  str(x).__contains__('Dropdown list')]
        for u in un:
            del nodes_data[i][u]

    # sets newly read investment options as new=True, sets read initially existing plants as new=False
    for i in config.ci: 
       
        for num, x in nodes_data[i].iterrows():
            if x['initial_existance'] == 1:
                x['new'] = False
            else:
                x['new'] = True

            nodes_data[i] = nodes_data[i].drop(labels=num)
            nodes_data[i] = nodes_data[i].append(x)

    return nodes_data


def validateExcelInput(dict):
    '''
    Should include more criteria, work in progress.
    Takes nodes from excel dict, returns dict after validation
    :param dict: nodes from excel
    :return: nodes from excel
    '''
    logging.info('Validating excel input sheet data.')

    # general checks
    tabs = [*dict]
    tabs.remove('timeseries')

    # check that no name ends in 4 digits (year gets added to the name later,
    for i in tabs:
        items = list(dict[i]['label'])
        for item in items:
            if item[-4:].isnumeric():
                raise ValueError(
                    f'Labels cannot end with four integers as this will lead to confusion \
                    once the year of investment is added to the year. \
                    Please change {item} in sheet {i} and start again.')
    logging.info('Labels from excel input are ok. Validation continues...')

    buses = list(dict['buses']['label'])

    # check for duplicate names
    for t in tabs:
        if any(dict[t].duplicated(subset=['label'])) == True: raise ValueError(
            f'Duplicate label entries in Excel input {tabs}')
    logging.info('No duplicate names found. Validation continues...')

    # check that the buses used for the investments e.g. input and output buses are included in the sheet 'buses'
    for i in tabs:
        if i == 'buses':
            continue

        lst_buses_labels = ['to', 'from', 'bus_1', 'bus_2', 'bus', 'from1', 'to1', 'from2', 'to2']
        sheet = dict[i]

        for label in lst_buses_labels:
            try:
                used_buses = list(dict[i][label])

                bus_issues = [bus for bus in used_buses if bus not in buses]

                if set(used_buses).issubset(buses):
                    continue
                elif all(pd.isna(bus_issues)):
                    continue

                else:
                    raise ValueError(
                        f'Please check sheet {i}. You have listed a bus in column \
                        {label} that is not included or set active in the bus list.')
            except KeyError:
                continue
    logging.info('All buses used as input or output have also been listed as a bus. \
                 Validation continues...')

    # check that demand buses have excess
    demands = list(dict['demand']['from'])
    bus_with_excess = list(dict['buses'].loc[dict['buses']['excess'] == 1]['label'])
    if not set(demands).issubset(bus_with_excess):
        raise ValueError(
            f'Please make sure that the buses listed as input for demands {demands} \
            have excess capacity enabled in the buses excel sheet.')
    logging.info('Checked that demand inputs have excess capability enabled. \
                 Validation continues...')

    # check that fixed flows exist for renewables and demand
    sheets = ['demand', 'renewables']
    fixed_flows = list(dict['timeseries'].columns)
    for sheet in sheets:
        df = dict[sheet]
        items = [item + '.fix' for item in list(df.loc[df['fixed'] == 1]['label'])]
        if not set(items).issubset(fixed_flows):
            raise ValueError(
                f'You have listed a fixed flow in sheet {sheet} that does not have a corresponding fixed flow in sheet timeseries.')
    logging.info('Checked availability of fixed flows. Validation continues...')

    # check that lca names have no special characters / < >
    lca_columns = ['var_env1', 'inv1', 'var_env2', 'inv2']
    sheets = [*dict]
    issue_chars = set(['/', '>', '<'])

    for sheet in sheets:
        if sheet in ['buses', 'timeseries']:
            continue

        for col in lca_columns:
            try:
                lca_items = list(dict[sheet][col])

                for item in lca_items:
                    if pd.isna(item):
                        continue
                    if 1 in [c in item for c in issue_chars]:
                        raise ValueError(
                            f'Please check excel sheet {sheet}, column {col} and rename --{item}-- in Excel \
                            and database as the following characters cause errors: {issue_chars}')
            except KeyError:
                continue
    logging.info('Checked for characters that cause issues in LCA dataset names. Validation continues...')

    logging.info('Successfully checked the scenario input table for the most common errors.')


def readWeightingNormalization(filename):
    '''
    Reads excel file for normalization and weighting
    :return: 2 dfs for normalization and weighting, dictionary of impact and unit
    '''

    # validate filenname
    if isinstance(filename, str) == True:
        if filename[-5:] != '.xlsx':
            raise ValueError('Make sure your filename for weighting and normalization ends in .xlsx')
        elif os.path.isfile(filename) == False:
            raise FileNotFoundError(f'No file found for weighting and normalization with name {filename}')
    else:
        raise TypeError(f'{filename} is not a valid filename; cannot read weighting and normalisation')

    # Validate objective inputs
    for i in config.objective:
        if not isinstance(i, str):
            raise TypeError('Please enter a valid objective')
        assert i in config.system_impacts_index, f'{i} not in list of allowed objectives'

    logging.info('Reading environmental weighting and normalization from Excel')

    weightEnvIn = pd.read_excel(filename, sheet_name='Weighting', index_col=1)

    # uses per person normalization if applicable
    if config.normalisation_per_person == True:
        normalizationEnvIn = pd.read_excel(filename, sheet_name='Normalisation_p_person', index_col=1)
    else:
        normalizationEnvIn = pd.read_excel(filename, sheet_name='Normalisation', index_col=1)
    logging.info("Environmental normalization and weighting successfully read")

    logging.info('Successfully read env weighting and normalization')

    return weightEnvIn, normalizationEnvIn


def getEcoinventDataOpenLCA(ps):
    '''
    openLCA access via olca client, (needs olca python lib installed)
    Fetch data for one product title and store in Excel file
    :param ps: product title
    :return: none
    '''

    client = olca.Client(8080)
    setup = olca.CalculationSetup()
    setup.calculation_type = olca.CalculationType.SIMPLE_CALCULATION
    setup.impact_method = client.find(olca.ImpactMethod, config.LCA_impact_method)

    setup.product_system = client.find(olca.ProductSystem, ps)
    setup.amount = 1.0 #check functionality

    # calculate the results and export it to an Excel file
    result = client.calculate(setup)
    client.excel_export(result, Path(f'LCA/{ps}.xlsx'))
    client.dispose(result)
    logging.debug(f'Completed update of {ps}')


def addLCAData(tech, weightEnv, normalizationEnv):
    '''
    Iterates through active elements and imports corresponding LCA data from 
    LCA folder or downloads them from openLCA (all impact categories).
    Multiplication with conversion factor to convert impacts to kW, kWh or m²
    if necessary. Division of investment impacts by lifetime.
    

    :param tech: nodes dict
    :param weightEnv: dfs for weighting, dictionary of impact and unit
    :param normalizatinEnv: dfs for normalization, dictionary of impact and unit
    :return: nodes dict with LCA data (normalized and weighted for multi-objectives)
    '''

    sheets = config.ci.copy()
    sheets.append('commodity_sources')
    sheets.append('buses')
    units = None

    adapted = tech.copy()
    ks = ['var_env1', 'var_env2', 'inv1', 'inv2', 'excess_env'] 

    # iterate through sheets from read-in Excel
    for i in sheets:

        new_nodes_data = pd.DataFrame(columns=tech[i].columns)

        # iterate through rows of excel sheet
        for x, row in tech[i].iterrows():

            # iterate through possible keys that contain LCA data
            data = {}
            for k in ks:
                try:
                    if not pd.isna(row[k]):
                        nm = row[k] #name of LCA product system used to calculate variable or investment env. impacts
                        
                        logging.info('Getting LCA Data for ' + row['label'] + ': ' + nm)

                        # determine if excel file already exists
                        if nm == 'empty':
                            get_file = False

                            if not os.path.isfile(f'LCA/{nm}.xlsx'):
                                raise ValueError(
                                    'Please make sure that you have a file called empty.xlsx in your LCA folder'
                                    'that has a value of 0 for all environmental factors.')

                        elif not os.path.isfile(f'LCA/{nm}.xlsx'):
                            get_file = True

                        else:
                            get_file = False

                        # set get_file to True so that all data will be updated through openLCA
                        if all([config.update_LCA_data, nm != 'empty']):
                            get_file = True

                        if get_file:
                            getEcoinventDataOpenLCA(nm)

                        df_new = pd.read_excel(f'LCA/{nm}.xlsx', sheet_name='Impacts', header=1, usecols="B:E",
                                               index_col=0)

                        if units == None:
                            units = dict(zip(df_new['Impact category'],
                                             df_new['Impact category'] + ' [' + df_new['Reference unit'] + ']'))

                            for item in list(weightEnv.columns):
                                if item in units.keys():
                                    continue
                                elif item in ['Indicator', 'Symbol']:
                                    continue
                                else:
                                    units[item] = item

                        df_new = pd.Series(df_new.Result)

                        # multiplies with conversion factor if applicable
                        if not pd.isna(row[f'{k}_conversion']):
                            logging.info('Multiplying with conversion factor')
                            df_new = df_new.multiply(row[f'{k}_conversion'])

                        # divides investment impacts by lifetime
                        if k in ['inv1', 'inv2']:
                            df_new = df_new.div(row['lifetime'])
                        
                        result = pd.DataFrame()

                        # apply weighting and normalization
                        indicators = [elem for elem in list(weightEnv.columns) if elem not in ['Indicator', 'Symbol']]

                        for indicator in indicators:
                            temp_result = df_new * weightEnv[indicator] / normalizationEnv[indicator]

                            result[indicator] = temp_result

                        data[f'{k}_LCA'] = result.sum(axis=0)
                
                except KeyError:
                    x = 'sth'

            for d in data:
                row[d] = data[d]

            # adds modified row to df
            new_nodes_data = new_nodes_data.append(row)

        # replaces original df with new df in excel node e.g. for transformers
        adapted[i] = new_nodes_data

    return adapted, units


def calcInvestAnnuity(tech): 
    '''
    Calculate the annuity for the investment
    :param tech: possible technologies for investment
    :return: possible technologies with information about annuity
    '''

    adapted = tech.copy()

    ci = config.ci.copy()
    
    for i in ci:
        new_nodes_data = pd.DataFrame(columns=tech[i].columns)

        for x, val in tech[i].iterrows():
            row = val
           
            name = row['label']

            # calculate annuity and use for ep_costs calculation
            try:
                if np.isnan(row['invest']):     
                
                    raise ValueError(f'No investment value for {name} provided.')
                
                if config.InvestWacc > 0:
                    ep_annuity = row['om'] + economics.annuity(capex=row['invest'],
                                                               n=row['lifetime'],
                                                               wacc=config.InvestWacc)
                else:
                    ep_annuity = row['invest'] / row['lifetime']
                
                row['annuity'] = ep_annuity

                logging.info('Calculated annuity for ' + row['label'] + ': ' + str(row['annuity'])) 
            except:
                logging.debug('No investment adaption necessary')

            new_nodes_data = new_nodes_data.append(row)

        adapted[i] = new_nodes_data

    return adapted


def determineEmissionFactors(tech):
    '''
    Determines the emission factors based on scenario file and the LCA data for technologies.
    :param tech: scenario file data extended with LCA data
    :return: tech, includes emission factors
    '''

    adapted = {}
    ks = ['var_env1', 'var_env2', 'inv1', 'inv2']

    for i in tech.keys():
        if i in ['buses', 'timeseries']:
            adapted[i] = tech[i].copy()
            continue

        df = pd.DataFrame(columns=tech[i].columns)
        for idx, row in tech[i].iterrows():

            if config.ef_fuel_based_only:
                try:
                    if row['fuel_based'] == 1:
                        fb = True
                    else:
                        fb = False
                except KeyError:
                    fb = False

            for k in ks:
                k_n = f'{k}_LCA'

                if config.emission_constraint == True:
                    try:
                        if k[:3] == 'var':

                            if config.ef_fuel_based_only:
                                if fb:
                                    row[f'{k}_EmissionFactor'] = row[k_n][config.ec_impact_category]

                                else:
                                    row[f'{k}_EmissionFactor'] = 0

                            else:
                                row[f'{k}_EmissionFactor'] = row[k_n][config.ec_impact_category]

                            logging.info(
                                row['label'] + f': Emission factor for {k}: ' + str(row[f'{k}_EmissionFactor']))
                        else:
                            logging.debug(row[
                                              'label'] + f': No emission factor (FF) for {k}. Investment EFs not yet working')
                    except:
                        logging.debug(row['label'] + f': No emission factor for {k}')

                else:
                    try:
                        if k[:3] == 'var':
                            row[f'{k}_EmissionFactor'] = 0
                            logging.info(
                                row['label'] + f': Emission factor for {k}: ' + str(row[f'{k}_EmissionFactor']))
                    except:
                        x = 'sth'
                        logging.debug(row['label'] + f': KeyError2 for {k}')

            df = df.append(row)

        adapted[i] = df

    return adapted


def getCostDict(tech):
    '''
    Get dictionary of investment costs and variable costs for possible technologies
    :param tech: possible technologies
    :return: {'invest': invest_dict, 'variable': variable_costs_dict}
    '''

    ci_cs = config.ci.copy()
    ci_cs.append('commodity_sources') 
    ci_cs.append('buses')

    inv_costs = {}
    var_costs = {}

    # iterate through sheets
    for i in ci_cs:

        # iterate through technologies
        for idx, row in tech[i].iterrows():

            try:
                inv_costs[row['label']] = row['annuity'] 
            except:
                logging.debug('No investment info for ' + row['label'])

            if i in ['commodity_sources', 'renewables']:
                var_costs[(row['label'], row['to'])] = row['variable costs']
            
            if i in ['buses']:
                var_costs[(row['label'], row['label'] + '_excess')] = row['excess_costs']
            
            if i == 'transformers_in':
                # outputs
                var_costs[(row['from1'], row['label'])] = row['var_from1_costs']
                var_costs[(row['label'], row['to1'])] = row['var_to1_costs']

                try:
                    if not pd.isna(row['to2']):
                        var_costs[(row['label'], row['to2'])] = row['var_to2_costs']
                except:
                    x = 'no second output'
                
            
            if i == 'transformers_out':
                # outputs
                var_costs[(row['from1'], row['label'])] = row['var_from1_costs']
                var_costs[(row['label'], row['to1'])] = row['var_to1_costs']

                try:
                    if not pd.isna(row['from2']):
                        var_costs[(row['from2'], row['label'])] = row['var_from2_costs']
                
                except:
                    x = 'no second input'
                    
            if i == 'storages':
                var_costs[(row['label'], row['bus'])] = row['variable output costs']
                var_costs[(row['bus'], row['label'])] = row['variable input costs']

    return {'invest': inv_costs, 'variable': var_costs}


def getEfDict(tech):
    '''
    Takes tech and returns dicitonary with emission factors only
    :param tech:
    :return: {'invest': inv_ef, 'variable': var_ef}
    '''

    inv_ef = {}
    var_ef = {}

    for i in tech.keys():

        if i in ['buses', 'timeseries']:
            continue

        for idx, row in tech[i].iterrows():

            if i in ['commodity_sources', 'renewables']:
                var_ef[(row['label'], row['to'])] = row['var_env1_EmissionFactor']

            if i == 'transformers_in':
                var_ef[(row['label'], row['to1'])] = row['var_env1_EmissionFactor']

                try:
                    if not pd.isna(row['to2']):
                        var_ef[(row['label'], row['to2'])] = row['var_env2_EmissionFactor']
                except:
                    x = 'no second output'
            
            if i == 'transformers_out':
                var_ef[(row['label'], row['to1'])] = row['var_env1_EmissionFactor']

                
            if i == 'storages':
                var_ef[(row['label'], row['bus'])] = 0
                var_ef[(row['bus'], row['label'])] = 0

    return {'invest': inv_ef, 'variable': var_ef}


def getEnvDict(tech):
    '''
    Takes possible technologies and returns dict for investment and variable impacts
    :param tech: possible techs with all corresponding impact factors
    :return: {'invest': invest_env, 'variable':variable_env_impacts}
    '''

    ci_cs = config.ci.copy()
    ci_cs.append('commodity_sources')
    ci_cs.append('buses')

    inv_env = {}
    var_env = {}

    for i in ci_cs:

        for idx, row in tech[i].iterrows():
            
            try:
                inv_env[row['label']] = row['inv1_LCA']
            except:
                logging.debug('No investment info for ' + row['label'])

            try:
                if not any(pd.isna(row['inv2_LCA'])):
                    inv_env[row['label']] = row['inv1_LCA'] + row['inv2_LCA']
            except:
                name = row['label']

            if i in ['commodity_sources', 'renewables']:
                var_env[(row['label'], row['to'])] = row['var_env1_LCA']
                
            if i in ['buses']:
                var_env[(row['label'], row['label'] + '_excess')] = row['excess_env_LCA']
                
            if i == 'transformers_in':
                # no input environmental impacts as applied in previous buses
                var_env[(row['label'], row['to1'])] = row['var_env1_LCA']
                
                try:
                    if not pd.isna(row['to2']):
                        var_env[(row['label'], row['to2'])] = row['var_env2_LCA']
                        
                except:
                    x = 'no second output'
                    
            if i == 'transformers_out':
                # no input environmental impacts as applied in previous buses
                var_env[(row['label'], row['to1'])] = row['var_env1_LCA']
                
                # no second output for transformers, where investment mode refers to output
                
    for var in [inv_env, var_env]:
        df = pd.DataFrame()
        for i in var.keys():
            df[i] = var[i]

        if var == inv_env:
            result = {'invest': df}
        elif var == var_env:
            result['variable'] = df

    return result


def saveFactorsForResultCalculation(tech, units, run_name, time):
    '''
    Get cost and env dict, merge to have one df for invest and one for variable
    :param tech:
    :return: factors dict for invest & variable
    '''

    logging.info('Saving cost and environmental factors for later result calculation.')

    # get cost dict for investments and variable costs
    cost_dic = getCostDict(tech)
   
    costs = {}
    costs['invest'] = pd.Series(data=cost_dic['invest'], name="Costs")
    costs['variable'] = pd.Series(data=cost_dic['variable'], name="Costs")

    # get lca data into env dict vor invest and variable environmental impacts
    env_dic = getEnvDict(tech)

    ef_dic = getEfDict(tech)
    ef_var = pd.Series(data=ef_dic['variable'], name="emission_factor")

    factors = env_dic.copy()

    for key in factors.keys():
        factors[key] = pd.concat([factors[key], costs[key].to_frame().T]).fillna(0)

    factors['invest'].loc['emission_factor'] = 0
    factors['variable'] = pd.concat([factors['variable'], ef_var.to_frame().T]).fillna(0)

    writer = pd.ExcelWriter(f'{run_name}\\files\\factors_{time}.xlsx', engine='xlsxwriter')

    for key in factors.keys():
        factors[key].to_excel(writer, sheet_name=key)

    writer.save()
    
    return factors


def calculateClimateNeutralEmissions(result):
    # get df of variable impacts only
    var = result.loc[result['type'] == 'variable']
   
    climate_neutral_emissions = var['emission_factor'].sum()

    return climate_neutral_emissions


def calculateEmissionConstraint2(tech, emission_goal_year):
    '''
    Takes information from sheet demand of scenario file to compute the maximum 
    value of emissions = emission constraint
    :param tech: as read in from scenario file
    :return: emission_constraint (float)
    '''

    logging.info('Calculating emission constraint')
    dem = tech['demand']

    emission_limit = 0
    for idx, row in dem.iterrows():

        ts = row['label'] + '.fix'
        energy_sum = tech['timeseries'][ts].sum()
        
        try:
            emission_factor = row[emission_goal_year]
            emission_limit += energy_sum * emission_factor
        except KeyError:
            emission_limit = None

    logging.info(
        f'The current year being calculated is {emission_goal_year - config.ec_horizon}. \
        The emission target used is for year {emission_goal_year}: {emission_limit} kg CO2-Eq/a')

    return emission_limit


def calculateEmissionGoals(tech, calc_years, climate_neutral):
    '''

    :param tech: technologies used in optimization
    :param calc_years: years used in the optimization
    :param climate_neutral: total value of kg co2-eq in the use phase that determine climate neutrality
    :return: df of years and associated emission goal values in total kg co2-eq per year (use phase only!)
    '''

    # create dataframe to save results in
    em_limits_years = pd.DataFrame(index=calc_years,
                                   columns=['calc year', 'emission goal year', 'political emission goal',
                                            'climate neutrality', 'use political goal'])

    # add first data to df
    em_limits_years['calc year'] = em_limits_years.index
    em_limits_years['emission goal year'] = em_limits_years + config.ec_horizon
    em_limits_years['climate neutrality'] = climate_neutral

    # calculate emission constraints for years that have values (=intermediary goals) in scenario file, sheet demand
    for year in calc_years:
        em_limits_years.at[year, 'political emission goal'] = calculateEmissionConstraint2(
            tech, year + config.ec_horizon)

    # get emission limits for those years without intermediary goals
    if em_limits_years['political emission goal'].to_list()[-1] == None:

        sy = sorted([x for x in tech['demand'].columns.to_list() if isinstance(x, int)])[
            -1]  # start year for next calculation
        cny = config.def_cn_year_climate_neutrality  # end year

        em_sy = calculateEmissionConstraint2(tech, sy)  # emissions for start year
        em_cny = climate_neutral  # emissions for climate neutrality

        # get the years that still need emission targets
        missing_years = em_limits_years.loc[pd.isna(em_limits_years['political emission goal'])].index.to_list()

        for year in missing_years:

            # determine emission goal year for the calculation year
            goal_year = em_limits_years.loc[year, 'emission goal year']

            # if the emission goal year is between the last intermediary goal year 
            # and the climate neutrality goal year, determine the goal based on a linear tightening of limits each year
            if goal_year < config.def_cn_year_climate_neutrality:
                
                b = (em_cny - (em_sy * sy / cny)) / (1 - cny / sy)
                em_year = (em_sy - b) / sy * goal_year + b
            else:
                em_year = climate_neutral

            em_limits_years.at[year, 'political emission goal'] = em_year

    # determine if political goal can be used, or if optimization will not work with these constraints
    for year in calc_years:
        if em_limits_years.loc[year, 'political emission goal'] >= climate_neutral:
            em_limits_years.at[year, 'use political goal'] = True
        else:
            em_limits_years.at[year, 'use political goal'] = False

    return em_limits_years

#######################################################
#------Functions used in optimizeForObjective()
#######################################################

def determineGoalForObj(objective):
    '''
    Sets ratio of financial and environmental cost
    :param objective: optimization objective
    :return: dict {"costs": goal_financial, "env": goal_environmental}
    '''

    # Validate function inputs
    if not isinstance(objective, str):
        raise TypeError('Please enter a valid objective')
    assert objective in config.system_impacts_index, f'{objective} not in list of allowed objectives'

    logging.info('*********** Determining Weight of Costs and Environmental Impacts *************')

    if objective == 'Costs':
        goal_financial = 1  # Costs (economical)
        goal_environmental = 0

    elif objective == 'EnvCosts':
        goal_financial = config.weight_cost_to_env
        goal_environmental = 1 - goal_financial
    
    elif objective == 'Equilibrium':
        goal_financial = config.weight_cost_to_env_equilibrium
        goal_environmental = 1 - goal_financial
    else:
        goal_financial = 0
        goal_environmental = 1

    logging.info('Successfully determined weighting of costs and environmental impacts')

    return {"costs": goal_financial, "env": goal_environmental}


def determineCfactorForSolver(objective):
    '''
    corrects value for solver, since errors occur with very small variables 
    for some impact indicators
    used only if normalisation is not per person
    :param objective: objective for optimization
    :return: c_weight
    '''

    # Validate function inputs
    if not isinstance(objective, str):
        raise TypeError('Please enter a valid objective')
    assert objective in config.system_impacts_index, f'{objective} not in list of allowed objectives'

    logging.info('Determining correction factor for solver')
    c_weight = 1
    
    if objective in {'human health - carcinogenic effects',
                       'human health - non-carcinogenic effects',
                       'human health - ozone layer depletion',
                       'human health - respiratory effects, inorganics'}:
        c_weight = 1e6
    elif objective == 'resources - minerals and metals':
        c_weight = 1e3
    elif objective in {'JRCII', 'EnvCosts', 'Equilibrium'}:
        if config.normalisation_per_person == True:
            c_weight = 1
        else:
            c_weight = 1e17
    elif objective in {'climate change - climate change biogenic', 
                       'climate change - climate change land use and land use change'}:
        c_weight = 100

    logging.info(f'Correction factor for solver: {c_weight}')

    return c_weight


def adaptEnvToObjective(tech, objective, goal_env, c_weight):
    '''
    Adapts environmental factors in tech to the optimization objective
    :param tech: nodes dict with LCA data (all values)
    :param weight: weighting determined for objective
    :param normalization: normalization determined for objective
    :param goal_env: environmental goal
    :param c_weight: correction factor for solver if LCA values are very small
    :return: nodes dict with environmental data as one value for multicriteria objectives
    '''
    
    sheets = config.ci.copy()
    sheets.append('commodity_sources')
    sheets.append('buses')

    # copy so that original df stays same
    adapted = tech.copy()
    # relevant keys for environmental data
    ks = ['var_env1', 'var_env2', 'inv1', 'inv2', 'excess_env']

    # iterate through sheets of possible investment classes in tech (in read from xlsx file)
    for i in sheets:

        new_nodes_data = pd.DataFrame(columns=tech[i].columns)
        if len(tech[i]) == 0:
            adapted[i] = tech[i]

        else:
            # iterate through rows of tech retrieved from scenario file
            for x in range(len(tech[i])):
                row = tech[i].iloc[x].copy()
                logging.info('Adjusting environmental impacts of ' + row[
                             'label'] + ' for weight and normalisation'
                            )

                
                for k in ks:
                    k_n = f'{k}_LCA'

                    try:
                        if isinstance(row[k_n], pd.Series):
                            if objective == 'Costs':
                                new = 0
                            else:
                                new = row[k_n][objective]

                            logging.info(row['label'] + ': Environmental impact included in optimization for ' + k_n + ': ' + str(
                                new))

                            row[k_n] = new * goal_env * c_weight

                            logging.info(row['label'] + ': Environmental result including goal_env (' + str(
                                        goal_env) + ') and c_factor (' + str(
                                        c_weight) + ') for ' + k + ': ' + str(row[k_n])
                                        )

                    except KeyError:
                        x = 'sth'

                # no investments required for commodity sources or excess (=buses)
                if i in ['commodity_sources', 'buses']:
                    new_nodes_data = new_nodes_data.append(row)

                # adapt investments, add ep_env if two investments required
                else:
                    try:
                        inv2 = pd.isna(row['inv2'])
                        if inv2 == False:
                            row['ep env'] = row['inv1_LCA'] + row['inv2_LCA']
                        if inv2 == True:
                            row['ep env'] = row['inv1_LCA']
                    except KeyError:
                        row['ep env'] = row['inv1_LCA']

                    new_nodes_data = new_nodes_data.append(row)

            adapted[i] = new_nodes_data

    return adapted


def adaptEnvToCNnoInvestment(tech_obj):
    '''
    Adatps environmental data to climate neutrality without an investment
    :param tech_obj:
    :return:
    '''

    new = {}
    for i in tech_obj:
        if i == 'timeseries':
            new[i] = tech_obj[i]
            continue

        df = pd.DataFrame()

        for idx, val in tech_obj[i].iterrows():
            nm = val['label']

            try:
                if val['ep env'] > 0:
                    val['ep env'] = 0
                    logging.debug(f'{nm}: Changed ep env to 0 for climate neutral calculation.')

            except KeyError:
                logging.debug(f'{nm}: No ep env invest CO2 emissions')

            df[idx] = val

        new[i] = df.transpose()

    return new


def adaptEnvToCNfuelBasedOnly(tech_obj):
    '''
    Adapts environmental factors to climate neutrality for fuel based setting
    '''
    new = {}
    for i in tech_obj:
        if i in ['buses', 'timeseries']:
            new[i] = tech_obj[i]
            continue

        df = pd.DataFrame()

        for idx, val in tech_obj[i].iterrows():

            nm = val['label']

            try:
                if val['fuel_based'] != 1:

                    for k in ['var_env1_LCA', 'var_env2_LCA']:
                    
                        try:
                            if val[k] > 0:
                                val[k] = 0
                                logging.info(f'{nm}: Changed {k} to 0 for climate neutral calculation.')
                        except:
                            logging.debug(f'{nm}: No value for {k}')

            except KeyError:
                logging.debug(f'{nm}: KeyError for fuel_based')

            df[idx] = val

        new[i] = df.transpose()

    return new


def adaptCostsToObjective(tech_obj, objective, goal_cost, c_weight):
    '''
    Takes tech_obj and adapts costs for goal and c_weight

    :param tech_obj: nodes dict, adapted for objective
    :param objective: objective
    :param goal_cost: weight of financial
    :param c_weight: correction factor for solver
    :return: nodes dict
    '''

    logging.info(f'Adapting financial inputs to {objective}')

    #   copy so that original df stays same
    adapted = tech_obj.copy()

    ci_cs = config.ci.copy()
    ci_cs.append('commodity_sources')
    

    if config.normalisation_per_person == True and objective in ['EnvCosts', 'Equilibrium']:
        normalization_cost = config.normalization_cost_gdp / config.normalization_person_population
        logging.info(f'Normalizing costs for {objective}') 
        
    else:
        if objective in ['EnvCosts', 'Equilibrium']:
            normalization_cost = config.normalization_cost_gdp
            logging.info(f'Normalizing costs for {objective}')
        else:
            normalization_cost = 1

    # iterate trough transformers, renewables and storages
    for i in ci_cs:

        new_nodes_data = pd.DataFrame(columns=tech_obj[i].columns)

        lst = []
        for k in list(tech_obj[i]):
            if k[-5:] in ['costs', 'Costs']:
                lst.append(str(k))
        if i in ['transformers_in', 'transformers_out', 'renewables', 'storages']: 
            lst.append('ep costs')

        # iterate through list of possible investments e.g. for transformers
        for x in range(len(tech_obj[i])):
            row = tech_obj[i].iloc[x].copy()
            name = row['label']

            try:
                
                if np.isnan(row['annuity']): 
                    raise ValueError(f'No investment value for {name} provided.')

                row['ep costs'] = row['annuity'] 
                               
            except:
                logging.debug('No investment adaption necessary')

            for k in lst:
                row[k] = row[k] * goal_cost * c_weight / normalization_cost

            new_nodes_data = new_nodes_data.append(row)

        adapted[i] = new_nodes_data

    logging.info('Successfully adapted financial inputs to objective')

    return adapted


def adaptTechObjToYear(pi, year):
    '''
    Takes original investment opportunities, as imported from excel and adapted to objective. Adapts for calculation year.

    :param pi: possible investments from excel import
    :param year: calculation year
    :return: dict of possible investments & timeseries
    '''

    logging.info(f'Adapting excel import investment opportunities to calculation year: {year}')

    #   copy so that original df stays same
    adapted = pi.copy()

    ci = config.ci.copy()

    # iterate through transformers, renewables and storages
    for i in ci:

        new_nodes_data = pd.DataFrame(columns=pi[i].columns)

        # iterate through list of possible investments e.g. for transformers, 
        # and change name to name_year and add end of life date
        for x in range(len(pi[i])):
            # copy row to ensure original df does not get changed
            row = pi[i].iloc[x].copy()
            name = row['label']

            # adapt data for new year
            row['label'] = row['label'].replace(name, f'{name}_{year}')
            row['eol'] = year + row['lifetime']

            new_nodes_data = new_nodes_data.append(row)

        adapted[i] = new_nodes_data

    return adapted


def adaptTimeseriesToYear(tech_obj, year, dtindex):
    '''
    Takes fixed timeseries, cuts out leap day if applicable
    :param tech_obj: adapted info about investments
    :param year: calculation year
    :param dtindex: datetimeindex for oemaof
    :return: adapted info (tech) with right datetimeindex for optimization
    '''

    df = tech_obj['timeseries'].copy()

    # drop leap day if applicable
    if not calendar.isleap(year):
        loc_m = df.loc[df['month'] == np.int64(2)]
        leap_day = loc_m[loc_m['day'] == np.int64(29)].index
        df = df.drop(leap_day)

    issue = len(df) - len(dtindex)
    if issue != 0:  # raise ValueError ('timeseries and df not the same length')
        dtindex = dtindex[:config.number_of_time_steps]
        df = df[:config.number_of_time_steps]

    df = df.set_index(dtindex)

    tech_obj[f'timeseries_{year}'] = df

    return tech_obj


def createOemofNodes(year, nd=None, buses=None):
   
    '''
    Create nodes (oemof objects) from node dict

    :param year: year used in calculation
    :param nd: dict of nodes as imported from scenario xlsx file
    :param buses: existing buses, if applicable
    :return: buses dictionary {label, oemof object} + list of created node objects
    '''

    logging.info('Creating nodes from possible elements')

    if not nd:
        raise ValueError("No nodes data provided.")

    nodes = []

    # Create Bus objects from buses table
    if buses != None:
        busd = buses

    if buses == None:
        busd = {}

        for i, x in nd["buses"].iterrows():
            if x["active"]:
                bus = solph.Bus(label=x["label"])
                nodes.append(bus)

                busd[x["label"]] = bus
                if x["excess"]:
                    nodes.append(
                        solph.Sink(
                            label=x["label"] + "_excess",
                            inputs={
                                busd[x["label"]]: solph.Flow(
                                    variable_costs=x["excess_costs"] + x["excess_env_LCA"]
                                )
                            },
                        )
                    )
                if x["shortage"]:
                    nodes.append(
                        solph.Source(
                            label=x["label"] + "_shortage",
                            outputs={
                                busd[x["label"]]: solph.Flow(
                                    variable_costs=x["shortage costs"]
                                )
                            },
                        )
                    )
                logging.debug(x['label'] + ' (bus) created')

        # Create Source objects from table 'commodity sources'
        for i, x in nd["commodity_sources"].iterrows():
           
            # determine if the element can be used or technology is not yet available
            if x["active"] and year - x['year_of_availability'] <= year and year + x[
                'end_of_availability']:   
                nodes.append(
                    solph.Source(
                        label=x["label"],
                        outputs={
                            busd[x["to"]]: solph.Flow(
                                nominal_value=x['nominal_value'],
                                summed_max=x["max_availability"],
                                variable_costs=x["variable costs"] + x["var_env1_LCA"],
                                emission_factor=x["var_env1_EmissionFactor"]
                            )
                        },
                    )
                )
                logging.debug(x['label'] + ' (commodity source) created')

        # Create Sink objects with fixed time series from 'demand' table
        for i, x in nd["demand"].iterrows():
            if year - x['year_start'] <= year and year + x['year_end'] > year:
                # set static inflow values
                inflow_args = {"nominal_value": x["nominal value"]}
    
                # get time series for node and parameter
                for col in nd[f"timeseries_{year}"].columns.values:
                    if col.split(".")[0] == x["label"]:
                        inflow_args[col.split(".")[1]] = nd[f"timeseries_{year}"][col]
          
                # create
                nodes.append(
                    solph.Sink(
                        label=x["label"],
                        inputs={busd[x["from"]]: solph.Flow(**inflow_args)},
                    )
                )
                logging.debug(x['label'] + ' (demand) created')

    # Create Source objects with fixed time series from 'renewables' table
    for i, x in nd["renewables"].iterrows():
        # determine if the element can be used or if end of life or availability has been exceeded or technology is not yet available
        use = True if x['eol'] > year and year - x['year_of_availability'] <= year and year + x[
            'end_of_availability'] > year else False
        name = x['label']

        if use == True:

            # if new investment, adds investment (investment and variable costs and impacts) parameters
            if x['new'] == True :
                outflow_args = {
                    'variable_costs': x["variable costs"] + x["var_env1_LCA"],
                    'emission_factor': x["var_env1_EmissionFactor"],
                    'investment': solph.Investment(
                        ep_costs = x['ep costs'] + x['ep env'],
                        area = x['area'],
                        maximum = x["max_capacity_invest"])
                    }

            # if existing renewable, take fixed flow as determined by previous calculation
            if x['new'] == False :
                outflow_args = {'nominal_value': x['initially_installed_capacity'],
                                'variable_costs': x["variable costs"] + x["var_env1_LCA"], 
                                'emission_factor': x["var_env1_EmissionFactor"]}

            # add fixed timeseries energy availability to outflow args
            for col in nd[f"timeseries_{year}"].columns.values:
                if col.split(".")[0] == name[:-5]: 
                    outflow_args[col.split(".")[1]] = nd[f"timeseries_{year}"][col] #fix
                    break

            nodes.append(
                solph.Source(
                    label=f'{name}',
                    outputs={busd[x["to"]]: solph.Flow(**outflow_args)})
            )

            logging.debug(x['label'] + ' (renewable) created')

    # Create Transformer objects from 'transformers_in' table where investment refers to input capacity
    for i, x in nd["transformers_in"].iterrows():

        # determine if the element can still be used or if end of life or availability 
        # has been exceeded or technology is not yet available
        use = True if x['eol'] > year and year - x['year_of_availability'] <= year and year + x[
            'end_of_availability'] > year else False
        name = x['label']

        # set static inflow values
        if use == True:

            # if existing transformer_in, set fixed max and nominal value
            if x['new'] == False:

                input_args = {busd[x["from1"]]: solph.Flow(max=x['initially_installed_capacity'],
                                                           nominal_value=1,
                                                           variable_costs=x["var_from1_costs"], 
                                                           )
                              }
                output_args = {busd[x["to1"]]: solph.Flow(emission_factor=x["var_env1_EmissionFactor"], 
                                                          variable_costs=x['var_to1_costs'] + x['var_env1_LCA'],
                                                          )
                               }
                conversion_facts = {busd[x["to1"]]: x["conversion_factor_t1"],
                                    busd[x["from1"]]: x["conversion_factor_f1"]}

                # add second output if it exists
                try:
                    output_args[busd[x["to2"]]] = solph.Flow(emission_factor=x["var_env2_EmissionFactor"], 
                                                             variable_costs=x['var_env2_LCA'],
                                                             )
                    conversion_facts[busd[x["to2"]]] = x["conversion_factor_t2"]
                except:
                    logging.info(x['label'] + ': No to2; existing investment')


            # if new investment, set investment parameters on input
            elif x['new'] == True :
                input_args = {busd[x["from1"]]: solph.Flow(variable_costs=x["var_from1_costs"],
                                                          investment=solph.Investment(
                                                                ep_costs=x['ep costs'] + x['ep env'],
                                                                maximum = x["max_capacity_invest"]
                                                                )
                                                           )}

                output_args = {busd[x["to1"]]: solph.Flow(
                    variable_costs=x['var_to1_costs'] + x['var_env1_LCA'],
                    emission_factor=x["var_env1_EmissionFactor"],
                                    )}

                conversion_facts = {busd[x["to1"]]: x["conversion_factor_t1"],
                                    busd[x["from1"]]: x["conversion_factor_f1"]}

                # add second output if it exists
                try:
                    output_args[busd[x["to2"]]] = solph.Flow(
                        variable_costs = x['var_env2_LCA'],
                        emission_factor=x["var_env2_EmissionFactor"],
                                                            )
                    conversion_facts[busd[x["to2"]]] = x["conversion_factor_t2"]
                except:
                    logging.info(x['label'] + ': No to2; new investment')

            # create
            nodes.append(
                solph.Transformer(
                    label=f'{name}',
                    inputs=input_args,
                    outputs=output_args,
                    conversion_factors=conversion_facts,
                )
            )

            logging.debug(x['label'] + ' (transformer) created')

 # Create Transformer objects form 'transformers_out' table where investment refers to output capacity
    for i, x in nd["transformers_out"].iterrows():

        # determine if the element can be used or if end of life or availability has been exceeded or technology is not yet available
        use = True if x['eol'] > year and year - x['year_of_availability'] <= year and year + x[
            'end_of_availability'] > year else False
        name = x['label']

        # set static inflow values
        if use == True:

            ###
            if x['fixed_cop'] == 1:
                try:
                    cf = nd[f'timeseries_{year}'][name[:-5] + '.cop']
                except KeyError:
                    logging.debug(
                        f'No fixed flow given for fixed COP of {name}, using constant given conversion factor: ' + str(
                            x["conversion_factor_t1"]))
                    cf = x["conversion_factor_t1"]
            else:
                cf = x["conversion_factor_t1"]

            # if existing transformer, set fixed max and nominal value
            if x['new'] == False:

                input_args = {busd[x["from1"]]: solph.Flow(variable_costs=x["var_from1_costs"])}
                output_args = {busd[x["to1"]]: solph.Flow(
                                                          max=x['initially_installed_capacity'],
                                                          nominal_value=1, 
                                                          emission_factor=x["var_env1_EmissionFactor"],
                                                          variable_costs=x['var_to1_costs'] + x['var_env1_LCA'], 
                                                          )}
                conversion_facts = {busd[x["to1"]]: cf,
                                    busd[x["from1"]]: x["conversion_factor_f1"]}

                # add second input if it exists
                try:
                    input_args[busd[x["from2"]]] = solph.Flow(variable_costs=x["var_from2_costs"])
                    conversion_facts[busd[x["from2"]]] = x["conversion_factor_f2"]  
                   
                except:
                    logging.info(x['label'] + ': No from2; existing investment')


            # if new investment, set investment parameters on output
            if x['new'] == True:
                input_args = {busd[x["from1"]]: solph.Flow(variable_costs=x["var_from1_costs"],
                                                           )
                              }

                output_args = {busd[x["to1"]]: solph.Flow(
                                                          variable_costs=x['var_to1_costs'] + x['var_env1_LCA'],
                                                          emission_factor=x["var_env1_EmissionFactor"],
                                                          investment = solph.Investment(
                                                              ep_costs=x['ep costs'] + x['ep env'],
                                                              maximum = x["max_capacity_invest"]
                                                                                        )
                                                         )}

                conversion_facts = {busd[x["to1"]]: cf,
                                    busd[x["from1"]]: 1} 
                

                # add second input if it exists
                try:
                    input_args[busd[x["from2"]]] = solph.Flow(
                        variable_costs = x['var_from2_costs'],
                    )
                    conversion_facts[busd[x["from2"]]] = x["conversion_factor_f2"]
                    
                except:
                    logging.info(x['label'] + ': No from2; new investment')

            # create
            nodes.append(
                solph.Transformer(
                    label=f'{name}',
                    inputs=input_args,
                    outputs=output_args,
                    conversion_factors=conversion_facts,
                )
            )

            logging.debug(x['label'] + '(transformer) created')
            
    for i, x in nd["storages"].iterrows():

        # determine if the element can still be used or if end of life has been exceeded
        use = True if x['eol'] > year and year - x['year_of_availability'] <= year and year + x[
            'end_of_availability'] > year else False
        name = x['label']

        # set static inflow values
        if use == True:

            # if new storage, set investment mode
            # dertermine if technology is available
            if x['new'] == True:
                nodes.append(
                    solph.components.GenericStorage(
                        label=x["label"],
                        inputs={
                            busd[x["bus"]]: solph.Flow(
                                variable_costs=x["variable input costs"],
                            )
                        },
                        outputs={
                            busd[x["bus"]]: solph.Flow(
                                variable_costs=x["variable output costs"],
                            )
                        },
                        loss_rate=x["capacity loss"],
                        initial_storage_level=x["initial capacity"],
                        balanced=bool(x['balanced']),
                        invest_relation_input_capacity=x['invest_relation_input_capacity'],
                        invest_relation_output_capacity=x['invest_relation_output_capacity'],
                        max_storage_level=x["capacity max"], 
                        min_storage_level=x["capacity min"], 
                        inflow_conversion_factor=x["efficiency inflow"],
                        outflow_conversion_factor=x["efficiency outflow"],
                        investment=solph.Investment(
                                                    ep_costs=x['ep costs'] + x['ep env'],
                                                    maximum = x["max_capacity_invest"]
                                                    )

                    ))

            # if existing storage, set fixed max and nominal value
            if x['new'] == False:
                nodes.append(
                    solph.components.GenericStorage(
                        label=x["label"],                        
                        inputs={
                                busd[x["bus"]]: solph.Flow(
                                nominal_value=x["initially_installed_capacity"] * x['invest_relation_input_capacity'],
                                                       variable_costs=x["variable input costs"],
                                                       )
                                },
                        outputs={
                                busd[x["bus"]]: solph.Flow( 
                                nominal_value=x["initially_installed_capacity"] * x['invest_relation_output_capacity'], 
                                                       variable_costs=x["variable output costs"], 
                                                       )
                                },
                        nominal_storage_capacity=x["initially_installed_capacity"],
                        balanced=bool(x['balanced']),
                        loss_rate=x["capacity loss"],
                        initial_storage_level=x["initial capacity"],
                        max_storage_level=x["capacity max"],
                        min_storage_level=x["capacity min"],
                        inflow_conversion_factor=x["efficiency inflow"],
                        outflow_conversion_factor=x["efficiency outflow"],
                    )
                )

            logging.debug(x['label'] + ' (storage) created')

    logging.info("The following objects have been created from excel sheet:")
    for n in nodes:
        oobj = str(type(n)).replace("<class 'oemof.solph.", "").replace("'>", "")
        logging.info(oobj + ":" + n.label)

    return busd, nodes
    


def getUsedTechnologies(esys, tech_obj_year):
    '''
    Goes through esystem results and returns Sources, Storages and Transformers that do not have flow = 0

    :param esys: energysystem after it was solved
    :param tech_obj_year: nodes dict, adapted for objective and for current year
    :return: dict of used technologogies for transformers, renewables, storages
    '''

    # investments excluding storage
    flows_non_storage = [x for x in esys.results["main"].keys() if not any(isinstance(n, solph.GenericStorage) for n in x)] 
    
    
    flows_with_value = [x for x in flows_non_storage if
                        esys.results["main"][x]['sequences']['flow'].max() > config.Invest_min_threshhold]
    
    flows_invest_in = [x for x in flows_with_value if x[1].label[-4:].isdigit()]
    flows_invest_out = [x for x in flows_with_value if x[0].label[-4:].isdigit()]
    
    # used technology for storages
    flows_storage = [x for x in esys.results["main"].keys() if x[1] is None]
    flows_storage = [x for x in flows_storage if
                     esys.results["main"][x]['sequences']['storage_content'].max() > config.Invest_min_threshhold]
   
    invest_dict = {
               'transformers_in': set([x for x in flows_invest_in if isinstance(x[1], solph.Transformer)]),
               'transformers_out': set([x for x in flows_invest_out if isinstance(x[0], solph.Transformer)]),
               'renewables': set([x for x in flows_invest_out if isinstance(x[0], solph.Source)]),
               'storages': set(flows_storage)}
    
    invests = {k: v for k, v in invest_dict.items() if len(v) > 0} 
    
    return invests


def exportInvestDecisions(esys, tech_obj_year, tech_obj_prev_year, year):
    '''

    :param esys: solved energysystem
    :param tech_obj_year: nodes dict, adapted for objective and for current year
    :param tech_obj_prev_year:  nodes dict, adapted for objective and for previous calc year, == None in first calc year
    :param year: current year
    :return: nodes dict
    '''

    logging.info('Saving investment decisions for next year')
    # determine the technologies that were used
    used_tech = getUsedTechnologies(esys, tech_obj_year)

    # get all flows from energysystem (without storage content)
    flows = [x for x in esys.results["main"].keys() if x[1] is not None]

    # go through transformers, storage etc.
    for x in used_tech.keys():
        logging.info(f'Iterating through {x}')
        # get list of items in nodes dict
        if tech_obj_prev_year != None:
            prev_nodes = tech_obj_prev_year[x]['label'].to_list()
            old = pd.DataFrame(columns=tech_obj_prev_year[x].columns)

        new_nodes = tech_obj_year[x]['label'].to_list() 
        new = pd.DataFrame(columns=tech_obj_year[x].columns) 

        # go through technologies within x (transformers, renewables, storages)
        for i in used_tech[x]:
            logging.info(f'Calculating info for {i}')
           # if new investment in this year, get corresponding row in nodes dict and append new df
            if x == 'transformers_in':
                if i[1].label in new_nodes:
                    pos = new_nodes.index(i[1].label) 
                    row = tech_obj_year[x].iloc[pos].copy() 
                    
                    if row['initial_existance'] == 0:
                        new = new.append(row)
                    
            elif x != 'transformer_in' and i[0].label in new_nodes:            
                pos = new_nodes.index(i[0].label) 
                row = tech_obj_year[x].iloc[pos].copy()

                if row['initial_existance'] == 0: 
                   new = new.append(row)

            new = new.drop_duplicates() 

            # if investment from previous year, get corresponding row in nodes dict and append new df
            if tech_obj_prev_year != None:
                
                if x == 'transformers_in':
                    if i[1].label in prev_nodes:
                        pos = prev_nodes.index(i[1].label)
                        row = tech_obj_prev_year[x].iloc[pos].copy()
                        if row['eol'] > year and row['initial_existance'] == 0:
                            old = old.append(row)

                elif x != 'transformer_in' and i[0].label in prev_nodes:
                    pos = prev_nodes.index(i[0].label)
                    row = tech_obj_prev_year[x].iloc[pos].copy()
                    if row['eol'] > year and row['initial_existance'] == 0:
                        old = old.append(row)


                    if x == 'storages':
                        flow = [x for x in used_tech['storages'] if x[0].label == row['label']]
                        last_storage_content = esys.results['main'][flow[0]]['sequences']['storage_content'].iloc[-1]                  
                        # sets initial capacity for the following year depending on the storage content of the last timestep                        
                        if row["initially_installed_capacity"] > 0 and last_storage_content > 0: 
                            row["initial capacity"] = last_storage_content / row[
                                    "initially_installed_capacity"]
                    
                old = old.drop_duplicates()

        # change status 'new' to false to ensure that oemof objects are not created based on investment values for next year
        new['new'] = False
        
        # for transformers: determine which flow is fixed and get max value, add to df
        df2 = pd.DataFrame(columns=new.columns)
              
        if x == 'transformers_in':
            nds = new['label'].to_list()
            logging.info(f'Looking at {x} for max values of input flows')

            # iterate through rows in transformers_in
            for i in nds:
                logging.info(f'Going through {i}')
                pos = nds.index(i)
                row = new.iloc[pos].copy()


                pot_flows = [x for x in flows if x[1].label == i]
                flow = [x for x in pot_flows if x[0].label == row['from1']]
 
                if row['initial_existance'] == 0:
                    row['initially_installed_capacity'] = esys.results['main'][flow[0]]['sequences'].max()['flow']
                
                df2 = df2.append(row)

            df2 = df2.drop_duplicates()  
            
        if x == 'transformers_out':
            nds = new['label'].to_list()
            logging.info(f'Looking at {x} for max values of output flows')

            # iterate through rows in transformers_out
            for i in nds:
                logging.info(f'Going through {i}')
                pos = nds.index(i) 
                row = new.iloc[pos].copy() 
                pot_flows = [x for x in flows if x[0].label == i]
                flow = [x for x in pot_flows if x[1].label == row['to1']] 

                if row['initial_existance'] == 0:
                    row['initially_installed_capacity'] =  esys.results['main'][flow[0]]['sequences'].max()['flow']
                
                df2 = df2.append(row)

            df2 = df2.drop_duplicates()  

        # for renewables, get flow for inv year and add it to timeseries
        if x == 'renewables':
            nds = new['label'].to_list()

            # iterate through rows in renewables
            for i in nds:
                logging.info(f'Going through {i}')
                pos = nds.index(i)
                row = new.iloc[pos].copy()
                
                flow = [x for x in used_tech['renewables'] if x[0].label == i]
                
                if row['initial_existance'] == 0:
                    try:
                        row['initially_installed_capacity'] = esys.results['main'][flow[0]]['scalars']['invest'] 
                    except KeyError:
                       
                        logging.debug(f'Error - sth went wrong: No investment for {x}, {i} found, \
                                      though it is an used technology. See script utils.exportInvestDecisions.')
                        sys.exit()                        
             
                df2 = df2.append(row)

        if x == 'storages':
            nds = new['label'].to_list()

            # iterate through rows in storages
            for i in nds:
                logging.info(f'Going through {i}')
                pos = nds.index(i)
                row = new.iloc[pos].copy()
                
                flow = [x for x in used_tech['storages'] if x[0].label == i]
                flow_ipts = (list(flow[0][0].inputs.items())[0][1].input, list(flow[0][0].inputs.items())[0][1].output)
                flow_opts = (list(flow[0][0].outputs.data.items())[0][1].input, list(flow[0][0].outputs.data.items())[0][1].output)

                try:                    
                    row["initially_installed_capacity"] = esys.results['main'][flow[0]]['scalars']['invest']
                    
                except KeyError:
                    x = 'Do nothing'
                    
                row['capacity inflow'] = esys.results['main'][flow_ipts]['sequences'].max()['flow']
                row['capacity outflow'] = esys.results['main'][flow_opts]['sequences'].max()['flow']
                
                last_storage_content = esys.results['main'][flow[0]]['sequences']['storage_content'].iloc[-1]
                if row["initially_installed_capacity"] > 0 and last_storage_content > 0: 
                    row["initial capacity"] = last_storage_content / row["initially_installed_capacity"]

                df2 = df2.append(row)

        if tech_obj_prev_year == None: old = pd.DataFrame(columns=df2.columns)

        
        tech_obj_year[x] = pd.concat([old, df2], sort=True)

    return tech_obj_year


def getResults(esys, prev_results=None):
    '''
    Takes energy system for new investments and prev_results for investments that were previously decided
    Returns dct of invests, sum of flows, and flows

    :param esys: solved energy system
    :param prev_results: results from previous year, if applicable
    :return: interim results, flows_to_tech: dict of flows and matching technology e.g. flow CHPg/el --> CHPg,
    :        tech_bus: definition of 'buses' for later consolidation of results
    '''

    logging.info('Getting oemof results to create list of investments')

    # get prev_results to right format
    if prev_results == None:
        prev_invests = {}
        flows = pd.DataFrame()

    else:
        prev_invests = prev_results['invest']
        flows = prev_results['flows']


    # create new variables
    variable = {}
    new_invests = {}
    flows_to_tech = {}
    tech_bus = {}
    new_flows = []
            
    # iterate through esys results
    for x in esys.results["main"].keys():
        logging.debug(str(x))
        if x[1] == None:
            storage_content = True 
        else:
            storage_content = False

        if storage_content:
             new_flows.append(esys.results['main'][x]['sequences'].rename(
                 columns={'storage_content': x[0].label + '/storage_content'}))

             technology = x[0].label
             
        else:
            new_flows.append(
                esys.results['main'][x]['sequences'].rename(columns={'flow': x[0].label + '/' + x[1].label}))
            # calculate sum of variable flows
            flow_sum = esys.results['main'][x]['sequences']['flow'].sum()
            if flow_sum > 0:
                variable[x[0].label + '/' + x[1].label] = flow_sum
            
            #definition of 'technology' for later consolidation of results
            technology = [item.label for item in x if not isinstance(item, solph.Bus)][0]
            flows_to_tech[x[0].label + '/' + x[1].label] = technology
            
            #definition of 'buses' for later consolidation of results
            bus = [item.label for item in x if isinstance(item, solph.Bus)][0]
            tech_bus[x[0].label + '/' + x[1].label] = bus

       
        if technology in prev_invests.keys():
        # if technology e.g. PV_year was already used in previous year (=previous investment), 
        # copy the invest from previous year to new year 
            new_invests[technology] = prev_invests[technology]

        # add new invests
        else:
            try:
                inv_value = esys.results['main'][x]['scalars']['invest']
                if any(isinstance(n, solph.GenericStorage) for n in x): 
                    if x[1] != None: 
                        continue 
                    
                    else:  
                        new_invests[x[0].label] = inv_value
                        
                elif isinstance(x[1], solph.Transformer): 
                    new_invests[x[1].label] = inv_value
                                    
                elif isinstance(x[0], solph.Transformer): 
                    new_invests[x[0].label] = inv_value
                     
                else:
                    new_invests[x[0].label] = inv_value 
                    
            except KeyError:

                try:
                    if not any([put.label[-4:].isnumeric() for put in x]):
                        continue

                    else:
                        logging.debug(str(x) + ': no investment value was passed')


                except AttributeError:
                    logging.debug(str(x) + ': Attribute error for storage component. No inv_value was passed')
           
    #delete zero values
    for key in list(new_invests.keys()):
        if new_invests[key] < 1E-7:
            del new_invests[key]
            
    df_new_flows = pd.concat(new_flows, axis=1, sort=True)

    flows = pd.concat([flows, df_new_flows], axis=0, sort=True).fillna(0)
    
    objective = esys.results['meta']['objective'] 

    return {'invest': new_invests, 'variable': variable, 'flows': flows, 'objective': objective}, flows_to_tech, tech_bus


def multiplyOemofResultsWithFactors(int_res, factors, year):
    '''
    Take the oemof results for investment and flows and muliply with cost and environmental factors
    :param int_res: intermediate results, dict (invest, variable, flows) = activity factors
    :param factors: emission/cost factors for multiplication (invest, variable)
    :return: result for year
    '''

    logging.info('Multiplying oemof results with cost and environmental factors')

    results = {}

    # iterate through ['invest', 'variable']
    for type in factors.keys():

        df = factors[type] 
        dct = int_res[type]

        result = pd.DataFrame()

        for ky in dct.keys():
            if type == 'invest':
                
                if int(ky[-4:]) == year:
                    nm = ky[:-5]  
                else: 
                    nm = None

            elif type == 'variable':
                lst = ky.split('/')
                nm = tuple([i[:-5] if i[-4:].isnumeric() else i for i in lst])

            try:
                value = df[nm] * dct[ky]

                result[ky] = value

            except KeyError:
                logging.info(f'No result for {ky} in def multiplyOemofResultsWithFactors')

        # Adapt envcosts & equilibrium
        if config.normalisation_per_person == True:
            normalization_cost = config.normalization_cost_gdp / config.normalization_person_population
        else:
            normalization_cost = config.normalization_cost_gdp
        
        try:
            result.at['EnvCosts'] = result.loc['EnvCosts'] * (1 - config.weight_cost_to_env) + result.loc[
                'Costs'] * config.weight_cost_to_env / normalization_cost
            result.at['Equilibrium'] = result.loc['Equilibrium'] * (1 - config.weight_cost_to_env_equilibrium) + result.loc[
                'Costs'] * config.weight_cost_to_env_equilibrium / normalization_cost
        except KeyError:
            logging.info('Could not calculate result for EnvCosts or Equilibrium.')
        
        result['Sum'] = result.sum(axis=1)
        
        results[type] = result
    
    results['Sum'] = results['invest']['Sum'] + results['variable']['Sum']
    results['objective'] = int_res['objective'] 
    
    return results


def consolidateAnnualResults(year, flows_to_tech, tech_bus, result_year, result_total_objective, tech_result_year, tech, def_cn):

    '''
    Takes results for year and appends them to previous years results.

    :param year: year of optimization
    :param flows_to_tech: dict matching flows to associate technology
    :param tech_bus: dict matching flows to associated buses (=counterpart of flows_to_tech)
    :param result_year: results for the given year of optimization
    :param result_total_objective: consolidated results for several years for given objective
    :param tech_result_year: oemof result of invests and flows
    :param tech: data adapted from scenario file
    :return: result for several years for given objective
    '''

    logging.info('Consolidating result from current year with previous results.')

    if config.aux_years:
        if (year + config.aux_year_steps - 1) <= config.end_year:
            step_for_aux_years = config.aux_year_steps
        else:
            step_for_aux_years = config.end_year - year + 1
        logging.info(f'Step for aux years: {step_for_aux_years}')
    else:
        step_for_aux_years = 1

    # get technical results = from oemof
    if result_total_objective == None:
        tech_results = pd.DataFrame()
    else:
        tech_results = result_total_objective['tech']

    # iterate through invest, variable
    for key in tech_result_year.keys():
        if key == 'flows' or key == 'objective':
            continue
        
        for item in tech_result_year[key]:
            # find the technology associated with investment e.g. CHP for CHP_2021
                        
            if key == 'invest':
                if int(item[-4:]) == year:
                    new_row = {'label': item, 'type': key, 'year': year, 'value': tech_result_year[key][item]}
                    new_row['technology'] = item[:-5]

                    tech_cat = {}
                    
                    for cat in config.ci:
                        for label in list(tech[cat]['label']):
                            tech_cat[label] = cat
                    
                    c = tech_cat[item[:-5]]
                                            
                    new_row['unit'] = tech[c].loc[tech[c][tech[c]['label'] == item[:-5]].index[0], 'unit']                                          
                    new_row['period end'] = year + step_for_aux_years - 1
                    
                   
                else:
                    continue           

            if key == 'variable':
                new_row = {'label': item, 'type': key, 'year': year, 'value': tech_result_year[key][item]}
    
                technology = flows_to_tech[item]
                if technology[-4:].isnumeric():
                    technology = technology[:-5]
                new_row['technology'] = technology
                bus = tech_bus[item]
                new_row['unit'] = tech['buses'].loc[tech['buses'][tech['buses']['label'] == bus].index[0], 'unit']
                new_row['period end'] = year + step_for_aux_years - 1 

                if def_cn == False:
                    
                    new_row['value'] *= step_for_aux_years
    
                    if config.emission_constraint:
                        new_row['emission horizon year'] = year + config.ec_horizon
                            
            tech_results = tech_results.append(new_row, ignore_index=True)
            

    # Get impact results for overview
    # if first year, set up the df for full results
    if result_total_objective == None:
        result_total_objective = {}

        result_overview = pd.DataFrame()
        result_overview.index.name = 'name'

    else:
        result_overview = result_total_objective['impact']
        

    # add investment to overview
    for ky in result_year.keys():
        if ky == 'objective' or ky == 'Sum': 
            continue

        res = result_year[ky].T # transpose dataframe
        res = res.drop('Sum') 
        
                    
        if def_cn == False:
            res *= step_for_aux_years
                
            if config.emission_constraint:
                new_row['emission horizon year'] = year + config.ec_horizon
        
  
        res['year'] = int(year)
        res['name'] = res.index.copy()
        res['type'] = ky 
        res['period end'] = year + step_for_aux_years - 1 
            
        if ky == 'variable':
            
            res['invest'] = None 

            for item in flows_to_tech.keys():
                try:
                    res.loc[item, 'invest'] = flows_to_tech[item]

                except KeyError:
                    print(f'{item} does not have associated tech')

        elif ky == 'invest':
            res['invest'] = res.index.copy()
            
            tech_cat = {}
            
            for cat in config.ci:
                for label in list(tech[cat]['label']):
                    tech_cat[label] = cat
            
            for item in list(res.index):
                
                c = tech_cat[item[:-5]]
                index = tech[c].index[(tech[c]['label'] == item[:-5])].tolist()
                res.loc[item, 'eol'] = int(tech[c].loc[index, 'lifetime']) + int(item[-4:])

        res['technology'] = None
        for item in list(res.index):
            
            if res.loc[item]['invest'][-4:].isnumeric():
                res.loc[item, 'technology'] = res.loc[item, 'invest'][:-5]
            else:
                res.loc[item, 'technology'] = res.loc[item, 'invest']

        res = res.dropna()
        res = res.reset_index(drop=True) 
        
        res['objective value'] = tech_result_year['objective'] * step_for_aux_years
        result_overview = result_overview.append(res)
        
    # Environmental impacts and costs are added for investments from previous years.      
    result_overview = result_overview.reset_index(drop=True)
    names = result_overview['name'].tolist() 
    names = [x for x in names if not "/" in x] #no flows
    names = list(set(names)) #remove duplicates
    
    for name in names:
        
        index = result_overview.index[(result_overview['name'] == name)].tolist()
        for i in index:
            y = result_overview.loc[i, 'year'] #type: float
            int_year = int(y)
            if year == int_year: #the investment year
                continue
            
            if int_year == int(name[-4:]):
            
                row = result_overview.iloc[i].copy()
                period_end = year + step_for_aux_years - 1
            
                if int(name[-4:]) + step_for_aux_years - 1 < period_end and year < int(row['eol']):
                    row['year'] = year
                    row['period end'] = year + step_for_aux_years - 1
                    row = row.to_frame().T
                    result_overview = pd.concat([result_overview, row])
                                           
    result_total_objective['impact'] = result_overview
    
    result_total_objective['tech'] = tech_results
    
    return result_total_objective


def exportNodesData(run_name, excel_nodes, nm, name):
    '''
    Auxiliary function to export nodes dict as xlsx file at any given point
    :param excel_nodes: nodes dict
    :param nm: name to save file
    :return: nothing
    '''

    logging.info(f'Exporting xlsx file for {nm}')
    writer = pd.ExcelWriter(f'{run_name}\\files\\{nm}_{name}.xlsx', engine='xlsxwriter')

    for key in excel_nodes.keys():
        if key in ['impact', 'tech'] or key[:4] == 'flow':
            excel_nodes[key].to_excel(writer, sheet_name=key, index=False)
       
        else:
            excel_nodes[key].to_excel(writer, sheet_name=key, index=True)

    writer.save()

