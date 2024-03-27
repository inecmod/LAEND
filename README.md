# LAEND
1. About LAEND

LAEND stands for "Life Cycle Assessment based ENergy Decision", which is based on the open source toolbox oemof (open energy model framework) and the life cycle assessment software openLCA and enables a coupled energy system analysis with environmental sustainability assessment and optimization. The method used is Life Cycle Assessment (LCA). The tool is written in the Python programming language and is based on oemof version 0.4.1 . In addition to the tools mentioned, it also uses other Python libraries. Parts of this version of LAEND were developed by Dorothee Birnkammer as part of her master thesis.
LAEND is primarily an extension of OEMOF. It incorporates environmental impacts over the entire life cycle of energy technologies into the optimization. In multi-objective optimization, the model minimizes the total impact resulting from the sum of weighted and normalized monetary and environmental impacts. Alternatively, LAEND optimizes for a single objective, such as global warming potential. The environmental impact data in LAEND is imported via a link to openLCA, a product system modeling tool for life cycle assessment. The necessary LCA data for the main renewable energy technologies and energy sources are already provided in LAEND as importable xlsx files. The provided data have been modeled using the ecoinvent LCA database . The EU Environmental Footprint (EF) method 2.0  integrated in this database is used as the impact assessment method. This method provides 16 impact indicators and corresponding normalization and weighting factors to form a weighted sum of the individual indicators. This aggregated value is used instead of costs in multi-criteria optimization.
LAEND is based on linear programming and determines an energy system with minimal impact. To account for the transition to renewable energy systems, LAEND uses a myopic optimization approach to map a multi-year period. Existing plants can be included using a brownfield approach . The myopic optimization consists of three steps: (1) mapping the existing energy system, (2) optimizing for a representative year representing one period of support years, and (3) transferring the results to the next representative year representing the next period of support years. For example, a period is five years, of which the first year is always optimized. These steps are repeated until the entire modeling period has been optimized. To optimize the representative year, the graph structure of the energy system is created from a set of investment options. Economic and environmental factors are assigned to all components via investment effects and variable effects. The energy system model is then optimized with perfect foresight for the first year of the period of support years. The solution consists of a set of selected technologies with corresponding installed capacities. This configuration is stored along with the hourly dispatch results. The size of the assets determined is carried over to the next year of optimization. For the second optimization year (e.g. the fifth year of the optimization horizon), the program creates a graph based on previous investments, provided that the technical life of the plant has not been exceeded. New investment opportunities are also added to the model. In the second optimization year, the optimization problem can decide whether to use the existing assets or to invest in new assets. Again, the results are available as a set of technologies with associated capacities and their utilization. The iterative process ends after the last year of the modeling period has been solved and the results have been saved.
For multi-criteria optimization, the weighted sum of total costs and environmental impacts, aggregated as the environmental footprint, is minimized in a 50:50 ratio (can be customized). Alternatively, only the aggregated environmental impacts, costs, or each of the 16 environmental indicators can be minimized.
Objective outcomes include investment and usage phase information. The investment data includes the facilities implemented and their capacity. The use phase data includes the sum of energy (carriers) consumed or delivered in each year. The optimization results in terms of newly built facilities and flows are multiplied by all impact factors to compile the results in terms of total cost or full LCA. If representative years are used, each value is multiplied by the number of years in the period represented by the optimized year (default is 5 years). This gives the total impact of an optimization to an objective over the entire model period.
As an option political emission reduction targets are handled with emission constraints, an adjustable decarbonization foresight horizon, and model-specific climate neutrality.
Utilizing weather data of typical meteorological years (TMY) to create fixed timeseries simplifies data research and preparation, as the program can derive most fixed timeseries from a single TMY file. Additionally, this ensures cohesive timeseries, as all weather-dependent load curves relate to the same data.

Helpful resources
- Book Optimization of Energy Supply Systems by Janet Nagel https://doi.org/10.1007/978-3-319-96355-6
- Oemof documentation at https://oemof-solph.readthedocs.io/en/v0.4.1/index.html
- Oemof github at https://github.com/oemof
- A former version of LAEND with perfect foresight: Tietze, I., Lazar, L., Hottenroth, H., Lewerenz, S. (2020): LAEND: A Model for Multi-Objective Investment Optimisation of Residential Quarters Considering Costs and Environmental Impacts. Energies 13: 614. doi:10.3390/en13030614.

2. Installing LAEND
	1. Install Python 3.7 or 3.8 (this might be done by installing Anaconda, see https://en.wikipedia.org/wiki/Anaconda_(Python_distribution))
	2. Create your virtual environment (e.g. 
https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/)
	3. Install oemof 0.4.1 (https://oemof-solph.readthedocs.io/en/v0.4.1/index.html; https://www.youtube.com/watch?v=eFvoM36_szM)
	4. Install a solver (LAEND assumes you use the CBC solver, though this can be changed in the config file)
	5. Fetch LAEND from GitHub (https://github.com/inecmod/laend)
	6. Install the python packages as listed in requirements.txt (https://docs.anaconda.com/free/navigator/tutorials/manage-packages/)

With the download of LAEND, a limited number of usable energy technologies are available. To include other technologies, new LCA data must be created. This requires the freely available openLCA software with a compatible ecoinvent license (for a fee) (more on creating LCA data in section 4.2).
You should now be ready to run LAEND. 

3. File Structure of LAEND
LAEND is separated into four main files. They are connected as follows:
* Main.py: Main file to run if you want to run LAEND
* Laend_module.py: Two most important functions get triggered here:
	o Main(): This function runs all preparations that are independent of the optimization objective
	o optimizeForObjective(): This function gets an optimization objective and then runs the myopic optimization
* utils.py: Document that has all calculations/functions
* Config.py: This file contains all settings. 

For further instructions for using LAEND see the documentation file in this repository.

The development of this version was carried out within the research project InPEQt - Integrated cost- and life-cycle-based planning of decentralized energy systems for energy and resource efficient neighborhood development. The project was funded by the German Federal Environmental Foundation.
