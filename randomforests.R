#' Code to run random forests using the package "spatialRF"
#' 
#' Use the random forest package for non-spatial random forests
#' 
#' Many advantages to using spatialRF. Mainly, a responsive package maintainer, 
#' the ability to test for spatial autocorrelation (with temporal to be added
#' at a later date), incorporate spatial smooth surfaces, easy to implement, and
#' a number of plotting options.

library(spatialRF)
library(randomForestExplainer)
library(randomForest)

#' This function trains and tunes a randomforest, and then evaluates that forest on 10 seperate training and testing folds. The
#' function uses 80% of the data for training and 20% for testing. The tuning aspect runs the random forests with 
#' different values of mtry (number of variables used in each split) and n (the number of forests made) until there is 
#' no longer a 0.01 increase in R2.
source("RF_funct.R") 

#'This function computes variable importance through permutation (predicts from the model
#'as the predictor variable is, randomizes the variable and then predicts again, calculates 
#'mean squared error between the actual response and the predicted values (for both as is
#'and randomized), and finally, takes the sqrt of the randomised MSE - as is MSE / as is MSE)
source("imp_perm.R")

#Load data
data <- read.csv("BTO_data.csv")

#List variables
Vars <-  c("slope_tavg", "slope_prec", "slope_tra", "urban", "crop", "forest", "shannon", "mean_elev", 
           "rich_1970")


#Random forests

#'Run a random forest with default parameters (for regression, different for classification): 
#'Number of variables tried at each split (mtry): 3
#'Number of trees (ntree): 500
#'Minimum node size (minimum size of terminal nodes, setting the number larger causes smaller
#'trees to be grown, and thus less time): 5
#'Max nodes (the maximum number of nodes the forest can have). If not specified, trees are grown
#'to maximum size possible: NULL

formula = as.formula(paste("mpd", "~" , paste(Vars, collapse = "+"), sep = ""))
mod <- randomForest::randomForest(formula, data = data, dependent.variable.name = "mpd", 
                                  predictor.variable.names = Vars,
                                  verbose = FALSE)

#Look at model outputs
mod

plot(mod) #Assess how the number of trees impacts the error
partialPlot(mod, pred.data = data, x.var = "slope_tavg") #Plot a partial plot

#Rerun with different number of trees
mod <- randomForest::randomForest(formula, data = data, dependent.variable.name = "mpd", 
                                  predictor.variable.names = Vars, ntree = 1000,
                                  verbose = FALSE)

plot(mod) #Assess again


#'Can use the predefined function to train the random forests in terms of ntree and mtry
#'It adds trees (500) until there is no improvement in R squared (evaluated on OOB data)
#'
#'The function also performs cross-validation by splitting the data into training and 
#'testing sets (10 different sets, 80% training and 20% testing).
 
out <- RF_funct(data = data, CP = 4, Response = "mpd", CoVars = Vars, 
                dirSave = paste0(getwd(), "/"))

#'Get variable importance values (permuted)
#'Outpath is where the models are stored 
#'Number of permutations is per cross validation fold
out <- importance_perm(outpath = paste0(getwd()), data = data, CP = 4, var = Vars, 
                       noperm = 1)






#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running models with spatialRF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set up the distance matrix first then run
fa <- data.frame(lat = data$ycoord, lng = data$xcoord)
dist_mat <- as.matrix(dist(fa, diag=T, upper=T))

#Run a model w/different distance classes to assess spatial autocorrelation
mod <- rf(data = data, dependent.variable.name = "mpd", 
          predictor.variable.names = Vars,
          distance.matrix = dist_mat, 
          distance.thresholds = c(10000, 30000), # Set the distance thresholds to test
          verbose = FALSE)

mod_tuned <- rf_tuning(model = mod, xy = data.frame(x = data$xcoord, y = data$ycoord), 
                       mtry = c(3,5), num.trees = c(500, 1000),
                       min.node.size = c(3,6), repetitions = 5,
                       training.fraction = 0.75, n.cores = 4, verbose = FALSE)
plot_tuning(mod_tuned) #Evaluate the tuning

#Number of options to plot
plot_importance(mod)
plot_residuals_diagnostics(mod)
plot_response_curves(mod) #Note the varying y axis
plot_response_surface(mod)
plot_moran(mod)#Notice the autocorrelation present at 30km

#Run a spatial model 
mod_spatial <- rf_spatial(model = mod, method = "mem.moran.sequential", verbose = FALSE,
                            n.cores = 8)

plot_importance(mod_spatial)#Notice the inclusion of spatial predictors
plot_moran(mod_spatial)

#Run tuning (tunes on spatial folds (number of repetitions))
mod_tuned <- rf_tuning(model = mod_spatial, xy = data.frame(x = data$xcoord, y = data$ycoord), 
                       mtry = c(3,5), num.trees = c(500, 1000),
                       min.node.size = c(3,6), repetitions = 5,
                       training.fraction = 0.75, n.cores = 4, verbose = FALSE)

#Returns that the model cannot be tuned, all approaches increase autocorrelation

#'Random forest is a stochastic algorithm that yields slightly different results on each run unless a random seed
#'is used. This particularity has implications for the interpretation of variable importance scores. For instance,
#'on one run of the RF the difference in importance scores could be merely due to chance. The function rf_repeat
#'(seen below the rf function) repeats a model execution and yields a distribution of importance scores across
#'executions. 

mod_repeat <- rf_repeat(mod_spatial, repetitions = 5)

plot_importance(mod_repeat)
plot_moran(mod_repeat)

#'So we have our model trained and tuned, have included spatial predictors (not-successfully it must be said..),
#'and we can get our importance/partial plots/performance.
#'
#'Next we need to evaluate performance across spatial folds and look a bit deeper at the importance values
#'OOB data is good to test on, but it is not completely independent of the forest. 
#'By creating spatially explicit folds we can evaluate the predictive performance of the model

mod_eval <- rf_evaluate(mod_spatial, repetitions = 10, training.fraction = 0.75, 
                        xy = data.frame(x = data$xcoord, y = data$ycoord))

plot_evaluation(mod_eval) #Predictive performance is poor...so very poor. 

#'Next we can use some helpful functions from the package "randomForestexplainer"

importance.df <- randomForestExplainer::measure_importance(mod_spatial,
                                                           measures = c("mean_min_depth", "no_of_nodes", 
                                                                              "times_a_root", "p_value"))
importance.df

#'From this we can see how many times the variable is a root, it's significance value (I find this the most useful
#'as it indicates whether the variable is included in more nodes than would be expected by chance), mean min depth 
#'is the average minimum tree depth at which the variable can be found, and no_of_nodes is the number of nodes in which
#'the variable was selected to make a split. 




