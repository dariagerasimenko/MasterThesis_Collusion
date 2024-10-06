This folder contains the followings files which are described

1. Six datasets.

1.1 DB_Collusion_Switzerland_Ticino_processed.csv
Collusive dataset from Switzerland Ticino. It has the columns:
Bid_value (Swiss Franc), Collusive_competitor, Consortium (group of companies which bid together as one), Tender, Date, Winner, Number_bids and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.2 DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv
Collusive dataset from Switzerland GR and See-Gaster. It has the columns:
Bid_value (Swiss Franc), Collusive_competitor, Contract_type, Tender, Date, Number_bids, Collusive_competitor_original, Winner and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.3 DB_Collusion_Japan_processed.csv
Collusive dataset from Japan. It has the columns:
Bid_value (Yen), Tender, Date, Site, Pre-Tender Estimate (PTE), Competitors, Name competitors, Winner, Collusive_competitor, Difference Bid/PTE, Number_bids, Period investigation by JFTC, Collusive_competitor_original and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.4 DB_Collusion_Italy_processed.csv
Collusive dataset from Italy. It has the columns:
Bid_value (Euro), Tender, Difference Bid/PTE, Winner, Competitors, Legal_entity_type (type of company), Site, Capital (money to registry the compan legally), Pre-Tender Estimate (PTE), Cartel_name, Collusive_competitor, Number_bids, Collusive_competitor_original and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.5 DB_Collusion_Brazil_processed.csv
Collusive dataset from Brazil. It has the columns:
Bid_value (Brazilian real), Tender, Bid (bid's number of the tender), Competitors, Difference Bid/PTE, Date, Site, Brazilian State, Pre-Tender Estimate (PTE), Collusive_competitor, Number_bids, Winner, Collusive_competitor_original and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.6 DB_Collusion_America_processed.csv
Collusive dataset from USA. It has the columns:
Bid_value (US Dollar), Tender, Date, Competitors, Bid_value_without_inflation (US Dollar), Winner, Bid_value_inflation_raw_milk_price_adjusted_bid (US Dollar), Collusive_competitor, only_cartels_bidding_on_contract, Number_bids, Collusive_competitor_original and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

1.7 DB_Collusion_All_processed.csv
Collusive dataset with all previous countries. It has the columns:
Bid_value (Euro, see exchange rates applied), Tender, Winner, Number_bids, Collusive_competitor, Dataset and the screening variables (CV, SPD, DIFFP, RD, KURT, SKEW, KSTEST)

Notes:
 - Date is timestamp format
 - Winner is 1 (the bid is the winner of the tender) or 0 (the bid is not the winner of the tender)
 - Competitors is the ID for each company of the dataset
 - Collusive_competitor is 1 (collusive) or 0 (not collusive)
 - Collusive_competitor_original is the original collusion without our treshold (see paper): 1 collusive, 0 not collusive
 - Exchange rates to Euros applied (early 2021): 1 Brazilian real (0.15€), 1 Yen (0.84€), 1 Swiss Franc (0.92€) and 1 US Dollar (0.0078€) 


2. Collusion_Detection_Code.py
It is the code in Python to analyse the collusive datasets and to generate the graphs and tables which are shown in the academic paper


3. requirements.txt
The file listing the necessary packages to run the .py
You can simply <pip install -r requirements.txt> and all of the program’s dependencies will be downloaded and installed.
