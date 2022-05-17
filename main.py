###############################################################################
# Generic imports, logging, global data frames
###############################################################################

# import libraries
import logging
from datetime import datetime
import multiprocessing as mp

# import python files
import laend_module
import config as config


if __name__ == "__main__":

    # start calculation and logging
    calc_start = datetime.now()
    run_name, time = config.createLogPathName()

    # run the non objective-specific function main
    tech, factors, emission_limits = laend_module.main(run_name, time)

    # create a multiprocessing pool
    if config.multiprocessing:
    # to limit the number of threads that run in parallel, add a number
    # for example: pool = mp.Pool(6)
        pool = mp.Pool()
    
        # start running objective-specific calculations in parallel
        for i in config.objective:
                pool.apply_async(laend_module.optimizeForObjective, args=(i, tech, factors, emission_limits, run_name, time))
    
    
        pool.close()
        pool.join()
    
    else: 
        for i in config.objective:
            laend_module.optimizeForObjective(i, tech, factors, emission_limits, run_name, time)
   
    # run the final result aggregation. Only works if all optimization problems led to a solution
    final = laend_module.combineResults(run_name, time)

    logging.info('calc time: ' + str(datetime.now() - calc_start))