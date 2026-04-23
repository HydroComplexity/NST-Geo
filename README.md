# NST-Geo

Summary: Rivers transport water and dissolved chemicals from landscapes to the oceans. The concentrations of these solutes are often temporally dynamic and respond strongly to changes in discharge as a result of multiscale water-landscape interactions. Accurate prediction of solute variability is challenging because it depends on both hydrological and biogeochemical processes resulting from heterogeneous catchment characteristics, such as soils, topography, vegetation, and geology. Forward process-based models often struggle to represent this complexity, while many machine learning approaches overlook spatial variability in landscape attributes. In this study, we develop a machine learning framework based on a Transformer architecture to predict river solute dynamics. The model integrates high-frequency streamflow and solute chemistry timeseries with spatio-temporal landscape attributes, enabling analysis of how catchment properties influence riverine chemical dynamics over time. Our NST-Geo model (non-stationary transformer with geo-spatial data) improves predictive performance compared to a baseline NST model and better captures solute concentration dynamics at several day prediction horizons. In addition to improved accuracy, the model reveals the landscape regions which are most informative to stream chemistry dynamics. These findings demonstrate how physically informed machine learning can advance the prediction and understanding of river biogeochemical dynamics, supporting improved water quality management under non-stationary environmental conditions.

File Overview:

1. NST_functions.py: Contains all the classes, functions for NST-Geo and NST model.
2. NST_model.py: To run all the individual augmentation in the NST and NST-Geo for USRB, Illinois, USA data.
3. NST_utils.py: All the utility functions for spatial data processing
4. plot_NST_mt.py: Plots all the results from USRB, Monticello, Illinois, USA data.

Python versions used:

1. python: 3.10.13
2. cuda: 12.4
3. pytorch: 1.13.0
4. numpy: 1.24.4
5. matplotlib: 3.7.2
6. pandas: 2.1.4
7. scipy: 1.8.0
8. scikit_learn: 1.3.0
9. seaborn: 0.13.2
10. tqdm: 4.65.0
11. pickle
