*Step-1: Import Cleaned Data and create dummy variables;
data road_construction;
infile "road_construction_cleaned.csv" firstobs=2 delimiter=',';
input price fair ratio status dist bid day len pla pb pe pm ps ptc sub;

/* Creating dummy variables for the 5 districts */
    district1 = (dist = 1);
	district2 = (dist = 2);
	district3 = (dist = 3);
	district4 = (dist = 4);
    *excludes the 5th value because if all above are 0, that handles this case;


run;

*STEP-2: Print dataset;
proc print;
title "road_construction with Dummy Vars";
run;

*STEP 3: Data Exploration (On all datapoints);

*3a) Five number Summary;
	*Only need top row for DV
TITLE "Descriptives of road-construction (full dataset)";
PROC MEANS N MIN Q1 MEDIAN Q3 MAX MEAN;
VAR price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*3b) Histogram;
	*Predict that we will need a log transformation
TITLE "Histogram of price road-construction (full dataset)";
*Generate histogram, overlay a normal distribution curve;
PROC UNIVARIATE normal;
VAR price;
histogram / normal (mu=est sigma=est);
RUN;

*3c) BOXPLOT;
TITLE "Boxplot - By Subcontractor Utilization";
*sort - by the text variable, non-numeric;
PROC SORT;
BY sub;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT price*sub;
RUN;

TITLE "Boxplot - By District";
*sort - by the text variable, non-numeric;
PROC SORT;
BY dist;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT price*dist;
RUN;

TITLE "Boxplot - By status";
*sort - by the text variable, non-numeric;
PROC SORT;
BY status;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT price*status;
RUN;

*FREQUENCY
*3d) SUB BREAKDOWN - Frequency;
TITLE "Frequency by Subcontractor Utilization";
PROC FREQ;
TABLES sub;
run;

*Status BREAKDOWN - Frequency;
TITLE "Frequency by Status of contract (1=fixed, 0=competetive)";
PROC FREQ;
TABLES status;
run;

*District BREAKDOWN - Frequency;
TITLE "Frequency by Districts (1-5)";
PROC FREQ;
TABLES dist;
run;


*3e) Scatterplot;
*MATRIX yvar xvar1 xvar2;
PROC SGSCATTER;
TITLE "Scatterplot Matrix for Price";
MATRIX price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*3f) RESIDUALS;
PROC REG;
TITLE "Residual Plots";
MODEL price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
plot student.*predicted.;
plot student.*(fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub);
plot npp.*student.;
RUN;


*STEP 4: RUN FULL MODEL
*a) Full Regression Model;
* P-val is under Parameter Estimates
	* Pr > absVal(t);
PROC REG;
title "Full Model(No Transformation)";
MODEL price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*a) Correlation Value/Coefficient;
*Will output a table, look for abs(0.9) values;
PROC CORR;
TITLE "Pearson Correlation Values";
VAR price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*b);
* run M2 model with Outliers/Influential Points - use original dataset;
	*Don't remove yet because we haven't transformed;
PROC REG DATA = road_construction;
MODEL price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub
		/influence r;
run;

*c);
*TRANSFORM Y-VARIABLE
Log Transformation - Only the dependent (y) variable;
DATA road_construction;	*Use original dataset that was already imported;
SET road_construction;	*Reset the original dataset;
TITLE "Price in log transformation";
*Set the PrizeMoney Column of the Golf dataset to a log transformation;
ln_price = log(price);	
RUN;

*Then need to add a proc print to display this;
PROC PRINT DATA = road_construction;
VAR ln_price;
RUN;

*Print entire dataset with changes;
PROC PRINT DATA = road_construction;
title "road_construction with log(price)";
RUN;


*///////////////////////////////////////////////////////////////////////////////////////////////////////
*NOW RESTART ALL OF THE DESCRIPTIVES AND DIAGNOSTICS TO GET INFORMATION ON THE LOG TRANSFORMED FULL MODEL
/////////////////////////////////////////////////////////////////////////////////////////////////////////*

*data = road_construction 
*Make all titles in ln form

*STEP 3: Data Exploration (On all datapoints);

*3a) Five number Summary;
TITLE "Descriptives of road_construction log(price)";
PROC MEANS N MIN Q1 MEDIAN Q3 MAX MEAN;
VAR ln_price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*3b) Histogram;
*Generate histogram, overlay a normal distribution curve;
PROC UNIVARIATE normal;
TITLE "Histogram of ln(price)";
VAR ln_price;
histogram / normal (mu=est sigma=est);
RUN;

*3c) BOXPLOT;
TITLE "Boxplot - By Subcontractor Utilization";
*sort - by the text variable, non-numeric;
PROC SORT;
BY sub;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT ln_price*sub;
RUN;

TITLE "Boxplot - By District";
*sort - by the text variable, non-numeric;
PROC SORT;
BY dist;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT ln_price*dist;
RUN;

TITLE "Boxplot - By status";
*sort - by the text variable, non-numeric;
PROC SORT;
BY status;
RUN;

*generate boxplot;
*Put dependent var on x-axis
* y-axis variable * x-axis variable;
PROC BOXPLOT;
PLOT ln_price*status;
RUN;

*FREQUENCY
*3d) SUB BREAKDOWN - Frequency;
TITLE "Frequency by Subcontractor Utilization";
PROC FREQ;
TABLES sub;
run;

*Status BREAKDOWN - Frequency;
TITLE "Frequency by Status of contract (1=fixed, 0=competetive)";
PROC FREQ;
TABLES status;
run;

*District BREAKDOWN - Frequency;
TITLE "Frequency by Districts (1-5)";
PROC FREQ;
TABLES dist;
run;


*3e) Scatterplot;
*MATRIX yvar xvar1 xvar2;
PROC SGSCATTER;
TITLE "Scatterplot Matrix for Price";
MATRIX ln_price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*3f) RESIDUALS;
PROC REG;
TITLE "ln_price Residual Plots";
MODEL ln_price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
plot student.*predicted.;
plot student.*(fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub);
plot npp.*student.;
RUN;


*STEP 4: RUN FULL MODEL
*a) Full Regression Model;
* P-val is under Parameter Estimates
	* Pr > absVal(t);
PROC REG;
title "Full Model(Log Transformation)";
MODEL ln_price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*a) Correlation Value/Coefficient;
*Will output a table, look for abs(0.9) values;
PROC CORR;
TITLE "Pearson Correlation Values";
VAR ln_price fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*b);
* run M2 model with Outliers/Influential Points - use original dataset;
	*Don't remove yet because we haven't transformed;
PROC REG DATA = road_construction;
title "Outliers/IPs";
MODEL ln_price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub
		/influence r;
run;

*////
Now ready for train/test split (code below)
*STEP 3: Split the data into training and test sets - 75/25;
* samprate = 75% of observations to be randomly selected for training set
* out = train defines new sas dataset for training/test sets;
proc surveyselect data = road_construction out=road_trainingSet seed= 2983409
				  samprate= 0.75 outall; *0.75 for a 75/25 trainTest split;
run;

*To check if selected field got created Test/Train samples;
	*If selected=1, part of train
	*If selected=0, part of test;
proc print;
title "Train/Test Split";
run;

*STEP 4: Create training set dataset
*ONCE YOU HAVE TRAIN SET
*Create a new y_variable to assign value ONLY for training set;
*This is done to tell the model to use only training set to build the model;
	*The model, use selected=1 for training/model selection;

*Overwrite existing data set;
data road_trainingSet;
	set road_trainingSet;
	if (selected = 1) then train_Y = ln_price; *Since LINEAR REGRESSION, we want to set the y to be the dep. var ;
run;


*Print only first 20 observations;
	*Ensures we got the above right;
proc print data=road_trainingSet (obs=20);
title "20 observations of Training Set (selected=1)";
run;

*CHECK FOR MULTICOLLINEARITY ON NEWLY CREATED TRAINING SET FULL MODEL

*STEP 4;
*a) Correlation Value/Coefficient;
	*Will output a table, look for abs(0.9) values;
PROC CORR data= road_trainingSet ;
TITLE "Pearson Correlation Values (training set)";
VAR train_Y fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub;
RUN;

*b);
	* run model with Outliers/Influential Points - use train dataset;
PROC REG data= road_trainingSet;
title "Outliers/Infulential Points, (training set)";
MODEL train_Y= fair ratio status district4 day len pla pm ptc
		/influence r vif stb;
run;

*STEP 6: 
	*From now on use road_trainingSet ONLY
*RUN MODEL SELECTION ON TRAIN SET ONLY;

	* Backward;
proc reg data= road_trainingSet;
title "Backward Selection";
model train_Y= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub
				/selection= backward rsquare;
run;

	* Stepwise;
proc reg data= road_trainingSet;
title "Stepwise Selection";
model train_Y= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub
					/selection= stepwise rsquare;
run;

	*Adj-R^2;
proc reg data= road_trainingSet;
title "Adj-R^2 Selection";
model train_Y= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub
					/selection= adjrsq;
run;


*6a) OBTAIN FINAL MODEL;
proc reg data= road_trainingSet;
title "Model after Selection Process (Chosen Backward)";
model train_Y= fair ratio status district4 day len pla pm ptc;
run;

*Run all diagnostics and assumptions again, step 4
	*Fix issues

*CHECK FOR MULTICOLLINEARITY ON FINAL MODEL

*STEP 4;
*a) Correlation Value/Coefficient;
	*Will output a table, look for abs(0.9) values;
PROC CORR data= road_trainingSet ;
TITLE "Pearson Correlation Values (training set)";
VAR train_Y fair ratio status district4 day len pla pm ptc;
RUN;

*b);
	* run model with Outliers/Influential Points - use train dataset;
PROC REG data= road_trainingSet;
title "Outliers/Infulential Points, (training set)";
MODEL train_Y= fair ratio status district4 day len pla pm ptc
		/influence r vif stb;
run;

*c) 
	* remove influential observations and outliers;
	* Overwrite test set;
title "Remove Influencial Points and Outliers";
data road_trainingSet_5out;		*writing into;
set road_trainingSet; *removing from;

if _n_ = 5 then delete;
run;

*Print to see the observations removed;
proc print data= road_trainingSet_5out;
title "obs #5 removed";
run;

*c);
*Run new model with obs 5 removed, see if any improvement;
	*Check for any more issues -NONE;
PROC REG data= road_trainingSet_5out;
title "Outliers/Infulential Points, (training set)";
MODEL train_Y= fair ratio status district4 day len pla pm ptc
		/influence r vif stb;
run;

*FROM NOW ON MUST USE
*data= road_trainingSet_5out/////////////////////////////

*STEP 7 After double chekcing diagnostics and assumptions on final model
	*If it's accurate, compute test performance

*//////////////////////////////////////////////////////////////////////////////////////////////////////
*Run FINAL MODEL built from training set for reference;
PROC REG data= road_trainingSet_5out;
title "FINAL MODEL";
MODEL train_Y= fair ratio status district4 day len pla pm ptc;
run;
*//////////////////////////////////////////////////////////////////////////////////////////////////////

*RUN RESIDUALS ON FINAL MODEL;
PROC REG data= road_trainingSet_5out;
TITLE "Residual Plots for Final Model (obs5 removed)";
MODEL train_Y= fair ratio status district4 day len pla pm ptc;
plot student.*predicted.;
plot student.*(fair ratio status district4 day len pla pm ptc);
plot npp.*student.;
run;


*STEP 8 -TEST PERFORMANCE

*Compute PREDICTIONS;
*a) Create predictions;
proc reg data= road_trainingSet_5out;
TITLE "Create PREDICTIONS";
model train_Y= fair ratio status district4 day len pla pm ptc;

*Period is indicating a null value, 
	*if it's null it belongs to the test set;
output out= pred (where = (train_Y= .)) p= yhat;
	*Not including confidence intervals;
run;

*Print Test Set with pred. probability;
	*yhat value is computed;
proc print data = pred;
title "Final Model with PREDICTIONS for test set only";
run;


*b) Summarize the results of cross-validation for the model;
*Compute y - yhat (MAE and RMSE)
*Compute abs(y - yhat)
	*Difference between actual and predicted;
data pred_summary;
title "Differenece between observed and predicted in Test Set";
set pred;
dif = ln_price - yhat; *difference between observed and predicted values in test set;
abs_dif = abs(dif); *Diffrence between Obs - predicted;
run;

	*Compute predictive statistics root mean square error (RMSE) and mean absoute error (MAE);
proc summary data= pred_summary;
var dif abs_dif;
output out = pred_stats std(dif)=rmse mean(abs_dif)=mae; *RMSE and MAE;
run;

proc print data = pred_stats;
title "Validation statistics for model";
run;


*c);
	*Compute correlation of observed and predicted values in test set;
proc corr data= pred;
title "Compute R^2 for test set";
var ln_price yhat;
run; * Compute R for test set (Use to compute R^2) and then CV-R^2;



*d);
	*Compute CV-R^2 for our sole model with the entire dataset
	*Asses how well the model can generalize across different datasets
*TRAIN SET;
*MUST USE FULL MODEL EQ FOR THIS;

	*Use the entire dataset, will establish new train/test datasets;
proc glmselect data= road_construction
	plots= (asePlot Criteria);
title "5-fold Cross Validation (STEPWISE selection)";
partition fraction (test= 0.25);
model ln_price= fair ratio status district1 district2 district3 district4 bid day len pla pb pe pm ps ptc sub/ 
			selection= stepwise(stop= CV) cvMethod=split(5)cvDetails=all;
run;


*9. Compare Models (ASE taken care of above);
	*Since only one model, comparing train set to test set
		*ASE is above
*Take all of the above information and explain it:
	/* Specifically explain step 8 as it compares the predictive power of both datasets on unseen data
		it will tell us which one performs better and can be used to create a better model
	- explain why I think train performs better than test
		

*EXTRA STEPS:
	Create a random prediction using the training data;
*Specific observation PREDICTION;
* compute predictions on new value;

* STEP 6.1: creates dataset with new value = predicted dataset;
data pred_value;
title "Compute Specific observation PREDICTION ";
input fair ratio status district4 day len pla pm ptc;
datalines;
586 1.35 1 0 250 3.8 33.9 1.6 8.5 
;
run;
proc print;
run;


* STEP 6.2: join new dataset with original(testing) dataset;
title "New observation joined to dataset";
data dataset_withPredVal; *Creating a NEW dataset by adding the single obs to the existing dataset;
set pred_value road_trainingSet_5out; *Addig the new obs to the existing dataset;
run;

proc print;
run;


* STEP 6.3: compute regression analysis and confidence interval for average estimate;
*p is for predicted value;
*clm: CI;
*cli: RI;
	*Only need screenshot of the top (obs 1) row;
proc reg data= dataset_withPredVal ;
title "Confidence interval for estimate";
model train_y= fair ratio status district4 day len pla pm ptc
			/p clm cli;
run;


*FINAL MODEL: TRANSFORM COEFFICIENTS;
*Run FINAL MODEL built from training set for reference;
PROC REG data= road_trainingSet_5out; *Trainging set with outliers removed;
title "FINAL MODEL";
MODEL train_Y= fair ratio status district4 day len pla pm ptc; * final model obtained through selection process;
run;

*done done done

