
# import general libraries
import os
import logging
from datetime import datetime
import pandas as pd

# import oemof libraries
import oemof.solph as solph
from oemof.tools import logger as oelog
from oemof.solph import (EnergySystem, Model, constraints)

# import module files
import utils as utils
import config as config

# this function runs first and is not specific to the objective
def main(run_name, time):

    ###############################################################################
    # Configure Logger
    ###############################################################################


    oelog.define_logging(
        logfile='laend.log',
        screen_level=logging.INFO,
        file_level=config.log_file_level,
        logpath= run_name + '\\logs',
        timed_rotating = {'when' : "s", 'interval':5} #creates new file every 5 seconds to avoid overly large files with write errors
    )

    # pass main starting info to logger
    utils.writeParametersLogger()



    ###############################################################################
    # Import and configure fixed flows (weather, demand, renewables)
    ###############################################################################


    ############### work with standard data ######################################

    # import typical meteorological year as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
    tmy, tmy_month_year = utils.compileTMY(config.filename_tmy)

    if config.update_heat_demand:

        utils.getHeatDemand(testmode=True, ann_demands_per_type=config.ann_demands_per_type, temperature=tmy['T2m'])  
        
        if config.separate_heat_water:
            utils.importFixedFlow(run_name, config.filename_th_demand, 'demand', config.varname_th_low, col_name='total_heat')
            utils.importFixedFlow(run_name, config.filename_th_demand, 'demand', config.varname_th_high, col_name='total_water')
        else:
            utils.importFixedFlow(run_name, config.filename_th_demand, 'demand', 'load_th', col_name='total')
   
    utils.createSolarCollectorFixedFlow(config.filename_solar_collector_high, tmy, run_name) if config.update_Solar_Collector_data == True else None
    
    utils.createSolarCollectorFixedFlow(config.filename_solar_collector_low, tmy, run_name) if config.update_Solar_Collector_data == True else None

    utils.createHeatpumpAirFixedCOP(config.varname_a_w_hp_low, config.hp_temp_low, tmy, run_name) if config.update_heatpump_a_w_cop == True else None

    utils.createHeatpumpAirFixedCOP(config.varname_a_w_hp_high, config.hp_temp_high, tmy, run_name) if config.update_heatpump_a_w_cop == True else None
    
    if config.update_electricity_demand:
        utils.importFixedFlow(run_name, config.filename_el_demand, 'demand', config.varname_el_demand, sum_mult_profiles=True)


    if config.update_pvgis_data:
        utils.createPvProfileForTMY(config.filename_pvgis, tmy_month_year)

        utils.importFixedFlow(run_name, 'in/pvgis_tmy.csv', 'renewables', config.varname_pv_1, col_name = 'P', conversion = 1/1000)
        utils.importFixedFlow(run_name, 'in/pvgis_tmy.csv', 'renewables', config.varname_pv_2, col_name='P', conversion=1 / 1000)


    # utils.createWindPowerPlantFixedFlow(config.varname_wind, tmy, run_name) if config.update_Wind_data == True else None

       
    ###############################################################################
    # Import setup data
    ###############################################################################

    # import excel sheet of possible investments/usable technologies
    tech = utils.importNodesFromExcel(f'{run_name}\\files\\{time}_{config.filename_configuration}')
    utils.validateExcelInput(tech)


    ###############################################################################
    # Import & configure LCA, cost data and emission constraints for later use
    ###############################################################################

    # read weight and normalization factors for environmental impacts
    weightEnv, normalizationEnv = utils.readWeightingNormalization(config.filename_weight_and_normalisation)

    # add LCA data to possible technologies
    tech, env_units = utils.addLCAData(tech, weightEnv, normalizationEnv)

    # determine the investment annuitiy of possible technologies
    tech = utils.calcInvestAnnuity(tech)

    # determine the emission factors used for the emission constraint
    tech = utils.determineEmissionFactors(tech)

    # save environmental impacts and cost factors for result calculation
    factors = utils.saveFactorsForResultCalculation(tech, env_units, run_name, time) #HH: run_name
    

    if config.emission_constraint:

        # run optimization for just one leap year to get best possible result for objective of climate change total
        result = optimizeForObjective(config.ec_impact_category, tech=tech, factors=factors, emission_limit=None, run_name=run_name, define_climate_neutral=True)

        # calculate climate neutrality as per optimization above
        climate_neutral = utils.calculateClimateNeutralEmissions(result)

        # get the years that are needed for the optimizations later
        years_for_calc = range(config.start_year, config.end_year + 1, config.InvestTimeSteps)

        # determine emission goals for each calc year for later optimizations
        emission_goals = utils.calculateEmissionGoals(tech, years_for_calc, climate_neutral=climate_neutral)

        # export emission targets
        emission_goals.to_excel(f'{run_name}\\files\\emission targets_{time}.xlsx')
        logging.debug(emission_goals)

    else:
        emission_goals = None



    return tech, factors, emission_goals


    ###############################################################################
    # Configure data for the optimization objective
    ###############################################################################

    #run function for each element in config.objective
   # if config.objective == []: raise ValueError('You have not selected an objective for the calculation.')





def optimizeForObjective(i, tech, factors, emission_limit, run_name, time, define_climate_neutral = False):

    # get admin things out of the way: logger, start time, where to save things etc.
    if not i.find('|') == -1:
        i_name = i.replace('|', ',')
    else: i_name = i

    if define_climate_neutral: i_name = 'climate neutrality'

    #configure oemof logger
    oelog.define_logging(
        logfile=f'laend_{i_name}.log',
        screen_level=logging.DEBUG,
        file_level=logging.DEBUG,
        logpath= run_name + '\\logs',
        timed_rotating = {'when' : "s", 'interval':5} #creates new file every 5 seconds to avoid overly large files with write errors
    )

    obj_time = datetime.now()
    logging.info(f'Starting calculation for {i} at {obj_time}')

    #determine goal and weighting of environment and cost factors
    goals = utils.determineGoalForObj(i)

    # determine correction factor for solver for env. goals where values are very small
    c_factor = utils.determineCfactorForSolver(i)

    # sum environmental impacts based on weight and normalization
    tech_obj = utils.adaptEnvToObjective(tech, i, goals['env'], c_factor)
    
    if define_climate_neutral:

        if config.def_cn_include_investment == False:
            tech_obj = utils.adaptEnvToCNnoInvestment(tech_obj)

        if config.def_cn_fuel_based_only:
            tech_obj = utils.adaptEnvToCNfuelBasedOnly(tech_obj)

    # calculate (annuity,?) normalize and multiply with goal and c_factor 
    tech_obj = utils.adaptCostsToObjective(tech_obj, i, goals['costs'], c_factor)



    ###############################################################################
    # Configure data for the year for given optimization objective
    ###############################################################################


    # start of myopic optimization for objective
    if define_climate_neutral:
        years_for_calc = [2024]
    else:
        years_for_calc = range(config.start_year, config.end_year + 1, config.InvestTimeSteps)

    for year in years_for_calc:

        # logging info about start of year calc
        year_time = datetime.now()
        logging.info(f'Starting calculation for {year} at {year_time}')

        # creating rolling emission constraint
        if config.emission_constraint and define_climate_neutral == False:
            if emission_limit.loc[year, 'use political goal']:
                emission_constraint = emission_limit.loc[year, 'political emission goal']
            else:
                emission_constraint = emission_limit.loc[year, 'climate neutrality']

        else:
            emission_constraint = 0



        if config.number_of_time_steps == None:
            datetime_index = pd.date_range(start=datetime(year, 1, 1, 00, 00), end= datetime(year, 12, 31, 23, 00), freq= config.granularity)
        else: datetime_index = pd.date_range(start = datetime(year, 1, 1, 00, 00), periods= config.number_of_time_steps, freq = config.granularity)

        # give investment possibilities new name = name_year
        tech_obj_year = utils.adaptTechObjToYear(tech_obj, year)

        # adapt fixed flows to year
        tech_obj_year = utils.adaptTimeseriesToYear(tech_obj_year, year, datetime_index)


        ###############################################################################
        # Create oemof nodes & other variables for myopic optimization
        ###############################################################################

        # initialisation of the energy system
        logging.info('Initializing energy system')


        if any([year == config.start_year, define_climate_neutral == True]):

            # in year one, no previous investments exist --> None
            tech_obj_prev_year = None

            # create oemof objects for the first year
            buses, new_nodes = utils.createOemofNodes(year, nd=tech_obj_year, buses= None)

            # create the energy system for the first year
            esys = EnergySystem(timeindex=datetime_index)

            # create variables required for later use
            int_res2 = None
            result_total_objective = None

        #elif year != config.start_year:
        else:
            # adapt timestamp to current year, take investments from previous year into account
            tech_obj_prev_year = utils.adaptTimeseriesToYear(tech_obj_prev_year, year, datetime_index)

            # create oemof nodes & buses from previous investments
            buses, prev_nodes = utils.createOemofNodes(year= year, nd = tech_obj_prev_year, buses = None)

            # recreate the energy system from the previous year
            old_esys = EnergySystem(timeindex=datetime_index) # Braucht es das hier? Es wird ein ganz neues Esys erstellt.
            old_esys.add(*prev_nodes)

            # pass the entities from the previous year to the energysystem for the current year
            esys = EnergySystem(timeindex=datetime_index, entities=old_esys.entities)

            # get buses from the created energy system (previous year)
            busd = {x.label: x for x in old_esys.entities if type(x) is solph.network.Bus}

            # create oemof nodes for possible new investments, pass existing buses to make sure investments match the buses
            new_buses, new_nodes = utils.createOemofNodes(year, nd=tech_obj_year, buses=busd)


        ###############################################################################
        # Create oemof energy system and solve
        ###############################################################################

        # add new nodes and flows to energy system
        logging.info('Adding nodes to the energy system')
        esys.add(*new_nodes)

        # creation of a least cost model from the energy system
        logging.info(f'Creating oemof model for {i} in year {year}')
        om = Model(esys)

        # set emission constraint (if config.emission_constraint = False, all emissions & emission_constraint = 0)
        if not emission_constraint == 0:

            # if calendar.isleap(year):
            #     emission_constraint = emission_constraint + emission_constraint*config.ec_leap_year_buffer
            logging.info(f'Optimization includes emission constraint: {config.emission_constraint}')
            logging.info(f'Optimization includes emission constraint: {emission_constraint}')
            constraints.emission_limit(om, limit=emission_constraint)

        logging.info(f'Oemof model for {i} in year {year} has been created')

        # solving the linear problem using the given solver
        logging.info(f'Solving the optimization problem for {i} in year {year}')
        ####################################################
        solved = False
        tries = 1

        while solved == False:
            try:
                om.solve(solver=config.solver)
                #HH               
                # logging.info('Saving oemof results for further processing')
                # esys.results["main"] = solph.processing.results(om)
                # esys.results["meta"] = solph.processing.meta_results(om)

                # logging.info('Storing the energy system with the results.')
                # esys.dump(filename=run_name + '\\oemof_dumps\\run_' + i_name + '_' + str(year) + '.oemof', dpath=run_name)

                solved = True
            except:
                logging.warning(f'Try #{tries} failed at solving the energy system. Changing emission constraints and trying again ')
                logging.info('New emission constraint: '+ str(emission_constraint*(1+config.ec_buffer)**tries))
                constraints.emission_limit(om, limit = emission_constraint*(1+config.ec_buffer)**tries)
                tries += 1

                if tries >= 3:
                    raise ValueError(f'The system for {year} could not be solved after 3 tries.')

        logging.info(f'Successfully solved the optimization problem for {i} in year {year}')



        ###############################################################################
        # Process results
        ###############################################################################

        # add results to the energysystem to make it possible to store them.
        logging.info('Processing results')
        if all([config.emission_constraint, define_climate_neutral == False]): logging.info('The system emissions for ' + config.ec_impact_category + ': ' + str(om.integral_limit_emission_factor()))
        
               
        logging.info('Storing the energy system with the results.')
        esys.dump(filename=run_name + '\\oemof_dumps\\run_' + i_name + '_' + str(year) + '.oemof', dpath=run_name)
        
        logging.info('Saving oemof results for further processing')
        esys.results["main"] = solph.processing.results(om)
        # logging.debug(esys.results["main"])
        esys.results["meta"] = solph.processing.meta_results(om)
        logging.info('Meta results for oemof calculation: ')
        logging.info(esys.results['meta'])

        # save the investments and elements that were used in this year for next years optimization
        tech_obj_prev_year = utils.exportInvestDecisions(esys, tech_obj_year, tech_obj_prev_year, year)
        
        # get the investment and flow results from energy system
        int_res2, flows_to_tech = utils.getResults(esys, int_res2)
       
        # ToDo: verify battery input/output issues #HH?
        
        int_res2['flows'].to_csv(f'{run_name}\\files\\Flows for {i_name}_{time}.csv')
        
        # multiply the oemof results with the cost and environmental factors
        result_year = utils.multiplyOemofResultsWithFactors(int_res2, factors, year)
        
        logging.debug('Objective value solver:')
        logging.debug(esys.results['meta']['objective'])
        logging.debug('Objective value calculated from results and factors')
        logging.debug(result_year['Sum'][i] * c_factor)
        
        result_total_objective = utils.consolidateAnnualResults(year, flows_to_tech, result_year, result_total_objective, int_res2, tech, define_climate_neutral)

        # export results to excel
        utils.exportNodesData(run_name, result_total_objective, f'Results for {i_name}', time)

        logging.info(f'Completed calculation for {year}, Calculation time: {datetime.now() - year_time}')

        if define_climate_neutral: return result_total_objective['impact']

    #return result_total_objective


def combineResults(run_name, time):

    logging.info('Aggregating all results into one file')

    df_impact = pd.DataFrame()
    df_tech = pd.DataFrame()
    df_obj_val = pd.DataFrame()

    for item in os.listdir(run_name + '\\files'):

        if item.__contains__('Results') and not item.__contains__('neutral') and item[-4:] == 'xlsx':
            xls = pd.ExcelFile(run_name + '\\files\\' + item)

            it_impact = xls.parse('impact')
            it_impact['objective'] = item[12:-25]

            it_tech = xls.parse('tech')
            it_tech['objective'] = item[12:-25]
            
            # it_obj_val = xls.parse('objective value')
            # it_obj_val['objective'] = item[12:-25]

            df_impact = pd.concat([df_impact, it_impact])
            df_tech = pd.concat([df_tech, it_tech])
            # df_obj_val = pd.concat([df_obj_val, it_obj_val])



    logging.info('Exporting Excel file for consolidated')
    writer = pd.ExcelWriter(f'{run_name}\\files\\Results total {time}.xlsx', engine='xlsxwriter')

    df_impact.to_excel(writer, sheet_name='impact', index=False)
    df_tech.to_excel(writer, sheet_name='tech', index=False)
    # df_obj_val.to_excel(writer, sheet_name='objective_val', index=False)

    writer.save()


