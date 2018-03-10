# Predictive Analytics on Diabetes dataset Using Naive Bayes

# Import data set
diabetes = read.csv("C:/Users/vajra/Documents/New R folder Excises,UPx/Diabetes1.csv")
# Find the shape of data set
dim(diabetes)
# Viewing first 6 rows
head(diabetes)
# Finding the data type of all the variables given
str(diabetes)

library(plyr)
library(ggplot2)

# changing the names of the columns into new ones using 'rename' function of 'plyr'and store
# Rename diabetes

diabetes = rename(diabetes,c("Pregnancies"="pregnancy","Glucose"="glucose",
                             "BloodPressure"="bp","SkinThickness"="skin",
                             "Insulin"="insulin","BMI"="bmi", 
                             "DiabetesPedigreeFunction"="pedigree",
                             "Age"="age","Outcome"="outcome"))
print(names(diabetes))

# Finding Quartile (1st QU,2nd QU or median,3rd QU or Max),in other words(25%,50%,75%,100%)
summary(diabetes)

# Missing value detection,Find NA values (Missing values) at the bottom per column
summary(diabetes) 
# Another approach is by sapply.Here function (x) means all variables given
diabetes.mv = sapply(diabetes,function(x)sum(is.na(x)))
diabetes.mv

# Missing Value Replacement
# Replacement of missing value of blood pressure with column median
# "replace" function is used for this.first argument (diabete$bp) is the name of dataset.
# 2nd argument (is.na(diabetes$bp)) is the variable having missing value,then we replaced
# the missing value of bp variable with the median of bp varaible "na.rm"=TRUE means the 
# complete values excluding the missing values of bp variable are taken for median computation.

diabetes$bp=replace(diabetes$bp,is.na(diabetes$bp),median(diabetes$bp,na.rm = TRUE))
# Replacement of missing value of Glucose with column median
diabetes$glucose=replace(diabetes$glucose,is.na(diabetes$glucose),
                         median(diabetes$glucose,na.rm = TRUE))
# Replacement of missing value of Pedigree with column median
diabetes$pedigree = replace(diabetes$pedigree,is.na(diabetes$pedigree),
                            median(diabetes$pedigree,na.rm =  TRUE))
# Replacement of missing value of  AGE with column median
diabetes$age = replace(diabetes$age,is.na(diabetes$age),median(diabetes$age,na.rm = TRUE))

# Now check again whether we have missing value or not
summary(diabetes)

# To make response variable (outcome) categorical,we need to use 'as.factor' function.
diabetes$outcome = as.factor(diabetes$outcome)

# Draw histogram for each variable to see number of zero values.
hist(diabetes$pregnancy)
hist(diabetes$glucose)
hist(diabetes$bp)
hist(diabetes$skin)
hist(diabetes$insulin)
hist(diabetes$bmi)
hist(diabetes$pedigree)
hist(diabetes$age)

# Replace zero values of few vaiables with column median
table(diabetes$glucose)
diabetes$glucose=replace(diabetes$glucose,diabetes$glucose == 0,
                         median(diabetes$glucose,na.rm = TRUE))
table(diabetes$bp)
diabetes$bp = replace(diabetes$bp,diabetes$bp == 0,median(diabetes$bp,na.rm = TRUE))

table(diabetes$skin)
diabetes$skin =replace(diabetes$skin,diabetes$skin == 0,median(diabetes$skin,na.rm = TRUE))

table(diabetes$insulin)
diabetes$insulin =replace(diabetes$insulin,diabetes$insulin ==0,
                          median(diabetes$insulin,na.rm = TRUE))

table(diabetes$bmi)
diabetes$bmi = replace(diabetes$bmi,diabetes$bmi==0,median(diabetes$bmi,na.rm = TRUE))

# No ZERO values
table(diabetes$pedigree)
table(diabetes$age)

# Check by finding minimum value of each column
min(diabetes$pregnancy)
min(diabetes$glucose)
min(diabetes$bp)
min(diabetes$skin)
min(diabetes$insulin)
min(diabetes$bmi)
min(diabetes$pedigree)
min(diabetes$age)

# Finding Quartile (1st QU,2nd QU or median,3rd QU or Max),in other words(25%,50%,75%,100%)
print(quantile(diabetes$pregnancy))
print(quantile(diabetes$glucose))
print(quantile(diabetes$bp))
print(quantile(diabetes$skin))
print(quantile(diabetes$insulin))
print(quantile(diabetes$bmi))
print(quantile(diabetes$age))

# Boxplot helps us to find or visualise if there is any outlier in the variable we check the 
# same to response variable (outcome)
ggplot(diabetes,aes(x=diabetes$outcome,y=diabetes$bp))+geom_boxplot()
# We check the presence of any outlier in pregnancy with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$pregnancy))+geom_boxplot()
# We check the presence of any outlier in glucose with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$glucose))+geom_boxplot()
# We check the presence of any outlier in skin with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$skin))+geom_boxplot()
# We check the presence of any outlier in insulin with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$insulin))+geom_boxplot()
# We check the presence of any outlier in bmi with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$bmi))+geom_boxplot()
# We check the presence of any outlier in pedigree with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$pedigree))+geom_boxplot()
# We check the presence of any outlier in age with respect to response variable (outcome)
ggplot(diabetes,aes(diabetes$outcome,diabetes$age))+geom_boxplot()

# Pair plot will shows the relationship between all the variables (except outcome) 
windows(7,7)+pairs(diabetes[,-9])

# we can visualise the variable to check if they follow normal 
# distribution using qqnorm & qqline
qqnorm(diabetes$age)
qqline(diabetes$age)
qqnorm(diabetes$glucose)
qqline(diabetes$glucose)

# Standardization of the data set.Rescaling dataset into one scale so that
# we dont want to rescale our target variable inside the square bracket -9
scaled_data = scale(diabetes[,-9])

# Check the standard deviation & mean of the standarized data
print(mean(scaled_data))
# Check the standard deviation of the standarized data
print(sd(scaled_data))

# now we append outcome variable to scaled_diabetes using dataframe function.
scaled_data = data.frame(scaled_data,diabetes$outcome)
head(scaled_data)

# Histogram
hist(scaled_data$glucose)
# Check whether mean become Zero or not to get the standard normalization
mean(scaled_data$glucose)
# Check whether standard deviation become 1 or not,
# above mean & standard deviation applicable to all varaibles
sd(scaled_data$glucose)


# DATA PARTITIONING
# "Caert" pacakage needs to be imported for the following functions.
library(caret)

# First argument scaled_diabetes is a vector of outcome or response that we have to define
# p = .75 means 75% data goes to training set.

Training_testing = createDataPartition(scaled_data$diabetes.outcome,p = .75,list = FALSE)

# [Training_testing,] it means we want first set of rows of train_testing
training = diabetes[Training_testing,]
# remaining amount data will go to testing
testing = diabetes[-Training_testing,]

# Check the dimension of training_set & testing_set
dim(training)
dim(testing)
head(training)

# Perform 10- fold cross validation,10 fold CV repeated 10 times.
fitcontrol = trainControl(method = "repetedcv",number = 10,repeats = 10)

# Train Naive Bayes model with ther training data set
library(naivebayes)

NBfit = train(outcome~.,data = training,method = "naive_bayes",trcontrol = fitcontrol)
print(NBfit)

# Predicting testing set with our model.[,-9] means
NB_Predict = predict(NBfit,testing[,-9])
confusionMatrix(NB_Predict,testing$outcome)
table(testing$outcome)
