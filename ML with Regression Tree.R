#----------relevant libraries
library(dplyr) 
library(ggplot2)
library(corrplot)
library(gridExtra) #for arranging all grids in a single page
library(caret)
library(rpart) #for model1
library(rpart.plot)
library(randomForest) #model2

#---------loading csv file
red <-read.csv("C:\\Users\\ps4pa\\Downloads\\winequality-red.csv", sep = ";")
str(red)
white <-read.csv("C:\\Users\\ps4pa\\Downloads\\winequality-white.csv", sep = ";") #separators for tabular display
str(white)

#---------data exploration
head(red)
head(white)
summary(red)
summary(white)
sum(is.na(red)) #checking for missing values
sum(is.na(white))

table(red$quality) #wine rating distribution, most are average while some are good quality?
table(white$quality)

#---------EDA
ggplot(red, aes(x = quality)) + geom_bar()
ggplot(white, aes(x = quality)) + geom_bar()

cor_red <- cor(red[, sapply(red, is.numeric)])
cor_white <- cor(white[, sapply(white, is.numeric)])
print(cor_red)
print(cor_white)
corrplot(cor_red, method = "color") #correlation matrix for red wine
corrplot(cor_white, method = "color") #correlation matrix for white wine

#---------boxplots: RED WINE
plots_list <- list() #empty list to store the plots

#forloop through all inputs except 'quality'
variable_names <- setdiff(names(red), "quality")
for (var_name in variable_names) {
  #creating a boxplot for each variable against red wine quality
  p <- ggplot(red, aes_string(x="factor(quality)", y=var_name)) +
    geom_boxplot(fill="red") +
    theme_minimal() +
    labs(title=paste(var_name, "vs Quality"), y=var_name, x="Quality")
  #adding the plot to the list
  plots_list[[var_name]] <- p
}

#arranging plots in a grid (ncol can/may be adjusted)
do.call(gridExtra::grid.arrange, c(plots_list, ncol=4))

#---------boxplots: WHITE WINE
plots_list <- list()
variable_names <- setdiff(names(white), "quality")
for (var_name in variable_names) {
  p <- ggplot(white, aes_string(x="factor(quality)", y=var_name)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title=paste(var_name, "vs quality"), y=var_name, x="Quality")
  plots_list[[var_name]] <- p
}

do.call(gridExtra::grid.arrange, c(plots_list, ncol=4))

#removing outliers:maybe later if needed

#---------combined datasets
red$type <- 'red'
white$type <- 'white'
wine_combined <- rbind(red, white)

#---------------------------------------------------------------------
#MODEL 1 CART

wine_combined$type <- as.factor(wine_combined$type)
wine_combined$quality <- as.numeric(wine_combined$quality)

#training and test sets
set.seed(123) #adjustable
index <- createDataPartition(wine_combined$quality, p = 0.8, list = FALSE)
training_data <- wine_combined[index, ]
testing_data <- wine_combined[-index, ]

#train the regression tree model
set.seed(456)
cart_model <- rpart(quality ~ ., data = training_data, method = "anova")

print(summary(cart_model))
predictions <- predict(cart_model, testing_data)

#RMSE and R-squared
rmse <- sqrt(mean((predictions - testing_data$quality)^2))
rsq <- cor(predictions, testing_data$quality)^2
cat("RMSE:", rmse, "\n")
cat("R-squared:", rsq, "\n")

#plotting the regression tree
rpart.plot(cart_model)

#---------------------------------------------------------------------
#MODEL 2 RANDOM FOREST

wine_combined$type <- as.factor(wine_combined$type)
wine_combined$quality <- as.numeric(wine_combined$quality)

#training and test sets
set.seed(123) # For reproducibility
index <- createDataPartition(wine_combined$quality, p = 0.8, list = FALSE)
training_data <- wine_combined[index, ]
testing_data <- wine_combined[-index, ]

#cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3, # Use 3 repeats for quicker computation
  summaryFunction = defaultSummary
)

#tune default mtry value for regression
default_mtry <- floor(ncol(training_data)/3)

#training model with default mtry value with cross-validation
set.seed(456) # For reproducibility
rf_model <- train(
  quality ~ .,
  data = training_data,
  method = "rf",
  trControl = fitControl,
  tuneGrid = data.frame(.mtry=default_mtry),
  metric = "RMSE"
)

#model summary to get the RMSE and R-squared values
print(rf_model)

predictions <- predict(rf_model, testing_data) #see predictions for test model

#RMSE and R-squared
rmse <- sqrt(mean((predictions - testing_data$quality)^2))
rsq <- cor(predictions, testing_data$quality)^2
cat("RMSE:", rmse, "\n")
cat("R-squared:", rsq, "\n")

rf_importance <- varImp(rf_model, scale = FALSE) #fetch variable importance
plot(rf_importance) #variable importance visualization

#Extra: confusion matrix and accuracy
rounded_predictions <- round(predictions)
factor_predictions <- as.factor(rounded_predictions)
factor_actual <- as.factor(testing_data$quality)
confusion_matrix <- confusionMatrix(factor_predictions, factor_actual)
print(confusion_matrix)
accuracy <- sum(rounded_predictions == test_set$quality) / length(test_set$quality)
cat("Accuracy of this model", accuracy, "\n")

#--------------------------------------------------------------------------