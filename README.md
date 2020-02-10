# Netflix_Recommander_System
Implementation of 5 methods of recommander system on Netflix data.
* Baseline Estimator ( *baseline_estimator.py* )
* Correlation Based Neighbourhood Model ( *correlation_based_neighbourhood_model.py* )
* Correlation Based Implicit Neighbourhood Model ( *correlation_based_implicit_neighbourhood_model.py* )
* SVD++ ( *svd_more_more.py* )
* Integrated Model ( *integrated_model.py* )

# How to use scripts
The default dataset path is "../Datasets" (relative path from the root of this repository), it can be modified in utils.py.

In this folder it must be a folder named "download" with the content of the archive that can be download here: 
https://archive.org/download/nf_prize_dataset.tar

The folder is not included in this repository due the size of the files.

Then you can run rating_compiler.py at first to create the different matrices from the files. Then you can run each algorithms.
