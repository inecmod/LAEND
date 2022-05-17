# LAEND_v031
1. About LAEND

LAEND stands for "Life Cycle Assessment based ENergy Decision", which is based on the open source toolbox oemof and the life cycle assessment software openLCA and enables a coupled energy system analysis with environmentally oriented sustainability assessment and optimization. The tool is developed in Python and uses different Python libraries besides the mentioned tools. LAEND is based on oemof 0.4.1. This version was created as part of the Master Thesis Extending the multi-objective model LAEND to account for sector coupled heat and electricity supply of residential quarters under consideration of decarbonization strategies by Dorothee Birnkammer. This documentation contains parts of the thesis.
LAEND is primarily an extension of oemof that incorporates environmental impacts in the optimization. For the multi-objective optimization, the model minimizes total impacts, which are the sum of weighted and normalized monetary and environmental impacts. Alternatively, LAEND optimizes for a single objective by setting irrelevant values to zero. Environmental impact data in LAEND is imported through a link to openLCA, a program for modeling life cycle systems. Data supplied stems mainly from ecoinvent 3.7. The environmental footprint (EF) method 2.0 is applied as an impact method. This method provides data for 19 indicators and corresponding normalization and weighting factors. LAEND launches the phase of the myopic optimization. The program creates an objective-specific energy system model and then optimizes to minimize impacts. This myopic optimization includes three steps: Create energy system, Optimize for one year and Pass results to next year. Thus the iterative process involves a single-year calculation that serves as the basis for the optimization in the subsequent year. Within an optimization for a single year, perfect foresight is assumed. These steps are repeated until the entire modeling period has been optimized. For the myopic optimization of Year 1, the graph structure of the energy system is created from a set of investment options. All components include information about investment impacts (invcost, invenv) and flows are parameterized with varcost and varenv, summarized as impact data. Next, the energy system model is optimized with perfect foresight for the first year of the modeling period. The solution is a set of selected technologies with corresponding installed capacities. This set is saved along with hourly dispatch results. The set of established installations is passed to the following year in the optimization, indicated by the green arrows. For the second optimization year, the program creates a graph based on previous investments, provided that the technical lifetime of the installation has not been exceeded. The resulting components contain no information about investment impacts but detail the existing capacity capmax. Flow parameters include varcost and varenv. Additionally, the graph is supplemented with new investment options. These are added with complete impact data. In the second year, the optimization problem can decide to employ existing installations or invest in new installations, if required. Again, the results present as a set of technologies with associated capacities. The iterative process stops after the last year in the modeling period has been solved for and results have been saved. If representative years are enabled, the calculation only optimizes the representative years and assumes that the previous year is equal to the preceding representative year.
Optimization outcomes are multiplied with all impact factors to compile results. The final result for one objective encompasses investment and use phase information. Investment data includes the implemented installation, its capacity, and the impact incurred over the modeling period. Use phase data contains the sum of energy required or supplied in each year and the corresponding variable impact. If representative years are used, each value is multiplied by the number of years in the period that one year represents. In sum, this provides the total impacts incurred by optimizing myopically for one objective over the entire modeling period.
As an option political emission reduction targets are handled with emission constraints, an adjustable decarbonization foresight horizon, and model-specific climate neutrality.

Helpful resources
- Book Optimization of Energy Supply Systems by Janet Nagel https://doi.org/10.1007/978-3-319-96355-6
- Oemof documentation at https://oemof-solph.readthedocs.io/en/v0.4.1/index.html
- Oemof github at https://github.com/oemof 

2. Installing LAEND v0.3.1
	1. Install Python 3.7 or 3.8 (this might be done by installing Anaconda, see https://en.wikipedia.org/wiki/Anaconda_(Python_distribution))
	2. Create your virtual environment (e.g. 
https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/)
	3. Install oemof 0.4.1 (https://oemof-solph.readthedocs.io/en/v0.4.1/index.html; https://www.youtube.com/watch?v=eFvoM36_szM)
	4. Install a solver (LAEND assumes you use the CBC solver, though this can be changed in the config file)
	5. Fetch LAEND from GitHub (https://github.com/inecmod/laendv031)
	6. Install the python packages as listed in requirements.txt
This version comes with a reduced number of usable energy technologies in order to be used without an ecoinvent licence for LCA data. Therefore, all required LCA data is available in the folder ~\LCA.
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
