****************************************************************
Prepared for Gabor's Data Analysis

Data Analysis for Business, Economics, and Policy
 by Gabor Bekes and  Gabor Kezdi
 Cambridge University Press 2021
 gabors-data-analysis.com 

Description of the 
bisnode dataset

used in case study 17A Predicting firm exit: Probability and classification

****************************************************************
Data source

Detailed company data from a middle-sized country in the European Union
All registered companies in 2005-2016 in three selected industries 
  (auto manufacturing, equipment manufacturing, hotels and restaurants)
This rich database was constructed for from multiple publicly available sources 
by Bisnode, a business data and anlytics company www.bisnode.com
for educational purposes

****************************************************************
Data access and copyright

You can use this dataset for educational purposes only.


****************************************************************
Raw data tables

*Raw data and cleaning codes will be added later.*

****************************************************************
Tidy data table

cs_bisnode_panel
 A company-year long format xt panel data table, 2005-2016 
 n = 287,829 observations (46,412 firms)
	(third as many companies in 2016 as in earlier years)
 ID variables	comp_id	(numerical) company identifier
		year	(calendar) year

 other important variables
		exit_year	year of exit (missing if still in business on 2016.12.31) 
		sales		sales in year (EUR)
		curr_assets	current assets at end of year (EUR)
		ind2		2-digit NACE industry code