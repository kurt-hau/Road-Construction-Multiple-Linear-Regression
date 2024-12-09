#Some preliminary data Cleaning done in R

# Import the data and view it
road_construction <- read.csv("~/Desktop/DSC Final project/DSC Final - Road Construction /road_construction.txt", sep="")
View(road_construction)

# Replace NA representaion '.' with NA in a specific column
road_construction$len <- replace(road_construction$len, road_construction$len == ".", NA)
View(road_construction)

#Delete rows with missing values
road_construction_noNA <- na.omit(road_construction)
View(road_construction_noNA)

#EXTRACTING MODIFIED DATASET
  #You want to include row names for same format
getwd() #Tells you where it will extract to
write.table(road_construction_noNA, file= "road_construction_cleaned.csv", row.names=F, sep= ",")


#EXTRA ANALYSIS
#Compute the amount of NA values
Missing_Count <- sum(is.na(road_construction$len))
Missing_Count

# Count the number of columns
num_columns <- ncol(road_construction)
num_columns

# Get a list of all the column names
column_names <- colnames(road_construction)
column_names