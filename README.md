# AUC_Boot-noBoot
AUC, standard error, and p-values for original data sets and bootstrapping in relation to article "Validating ShallowHRD for Clinical Use: Correlation with HRDetect in Familial Breast Tumors".


Contains 2 scripts: 

Script 1: 
Conducts bootstrapping for each of the three data sets in the article to determine AUC and standard errors, and compares this to 3 set AUC-values (returns p-values). 

Script 2: 
Does not conduct bootstrapping, and calculates the AUC directly from the datasets. 


#If used

If you use this, please cite: 
Adel Jensen, L., Baekgaard, C.H., Larsen, M.B. et al. Preliminary evaluation of ShallowHRD performance compared to HRDetect in familial breast cancer tumors. Sci Rep 15, 29442 (2025). https://doi.org/10.1038/s41598-025-14122-9

#Utilizing 
Scipy (Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2)
Scikit-learn (Pedregosa F, Varoquaux, Ga"el, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine learning in Python. Journal of machine learning research. 2011;12(Oct):2825–30.)

