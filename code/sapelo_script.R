# shared_lib_path <- "/work/crss8030/instructor_data/shared_R_libs"

#class_packages <- c("tidymodels", "tidyverse", "vip", "ranger", "finetune", "parsnip", "reticulate", "xgboost", "doParallel", "lme4", "here")

#install.packages(class_packages, lib = shared_lib_path)

# .libPaths(c("/work/crss8030/instructor_data/shared_R_libs", .libPaths()))

#install.packages("xgboost") #new pacakage

library(tidymodels)   # Core framework for modeling (includes recipes, workflows, parsnip, etc.)
library(finetune)     # Additional tuning strategies (e.g., racing, ANOVA-based tuning)
library(vip)          # For plotting variable importance from fitted models
library(xgboost)      # XGBoost implementation in R
library(ranger)       # Fast implementation of Random Forests
library(tidyverse)    # Data wrangling and visualization
library(doParallel)   # For parallel computing (useful during resampling/tuning)
library(lme4)         # The matrix package and other mathmatical computation for xgboost
library(here)         # Gives the absolute path starting from Rproj, works well with different scripts (R, .rmd or .qmd)
library(quarto)       # helps to export the qmd script to R script
#library(caret)       # Other great library for Machine Learning 

here::here()

# read using here package
weather <- read_csv() %>% 
  # separate site to extract the state abbrevations only
  
  # relocate the separated or newly created colnames 
  
  # remove the unwanted column
  dplyr::select(-city) %>% 
  mutate(year = as.factor(year))

set.seed(931735) # Setting seed to get reproducible results 

weather_split <- initial_split(
  weather, 
  prop = .7, # proption of split same as previous codes
  strata = strength_gtex  # Stratify by target variable
  )

weather_split

weather_train <- training(weather_split)  # 70% of data
weather_train #This is your traing data frame

weather_test <- testing(weather_split)    # 30% of data
weather_test

density_plot <- ggplot() +
  geom_density(data = weather_train, 
               aes(x = strength_gtex),
               color = "red") +
  geom_density(data = weather_test, 
               aes(x = strength_gtex),
               color = "blue") 

# lets print the undergoing process in the console, (important when running in the terminal later on)
cat(paste0("Saving...... density plot of train and test data"))

ggsave(plot = density_plot, 
       path = here("output", "png"),
       filename = "density_plot_test_train.png",
       height = 6,
       width = 9,
       dpi = 600)
  

cat("\nCreate recipe for data preprocessing\n")

weather_recipe <- recipe(strength_gtex ~ ., data = weather_train) %>% # Remove identifier columns and months not in growing season
  step_rm(
    matches("Jan|Feb|Mar|Apr|Nov|Dec")  # Remove non-growing season months
  )
  # updating the role of the 2 columns as ID, so they arenot used as predictors
  

weather_recipe

# Prep the recipe to estimate any required statistics
weather_prep <- weather_recipe %>% 
  prep()

# Examine preprocessing steps
weather_prep

# helps to see all the available hyperparameter we can tune for a model

# helps to see all the models that we can run with the package parsnip


xgb_spec <- #Specifying XgBoost as our model type, asking to tune the hyperparameters
  boost_tree(
   # Total number of boosting iterations
    trees = tune(),
         # Maximum depth of each tree
    tree_depth = tune(),
             # Minimum samples required to split a node
    min_n = tune(),
        # Step size shrinkage for each boosting step
    learn_rate = tune()
      ) %>%
        #specify engine 
  set_engine("xgboost") %>%
       # Set to mode
  set_mode("regression")
  
xgb_spec


set.seed(235) #34549

resampling_foldcv <- vfold_cv(weather_train, # Create 10-fold cross-validation resampling object from training data
                              v = 10)

# Create leave one year out cv object from the sampling data


# Create leave one location out cv object from the sampling data


set.seed(12345)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  learn_rate(),
  trees(),
  size = 20
)

xgb_grid

ggplot(data = xgb_grid,
       aes(x = tree_depth, 
           y = min_n)) +
  geom_point(aes(color = factor(learn_rate),
                 size = trees),
             alpha = .5,
             show.legend = FALSE)

# Detect Cores for Slurm (Cluster - Sapelo2)

# lets look into our current environment to see the available cores
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))

if (is.na(n_cores)) n_cores <- parallel::detectCores() - 1

# Start the Cluster
cl <- makePSOCKcluster(n_cores)

registerDoParallel(cl)

cat(paste0("\nFound and registered ", n_cores, " cores to work with\n"))



set.seed(76544)

# Create the list of CV techniques so we can loop through them


# Create a emty list to store the results from the loop

# Create our loop



stopCluster(cl)


library(dplyr)

# Create a dataframe structure from the list obtained after running the loop
results_df <-  
  # Collect the metrices for each CV techniques using map function


# bind all the metrices together so to select the best performing one
all_metrices <- do.call(bind_rows, results_df$metrices)

# Automating to pull the best method out of 3 we ran

best_method <- all_metrices %>%
  

# Getting the metrice (hyperparameter values of the best performing CV)
best_cv_object <- all_metrices %>%
  filter(method == best_method) %>%
  pull(diff_cv) %>% 
  first()
  

# Best RMSE
best_rmse <- best_cv_object %>% 
      select_best(metric = "rmse")%>% 
  mutate(source = "best_rmse")

best_rmse

# Based on greatest R2
best_r2 <- best_cv_object %>% 
  select_best(metric = "rsq")%>% 
  mutate(source = "best_r2")

best_r2

best_rmse %>% 
  bind_rows(best_rmse, 
            best_r2) %>%
  dplyr::select(source, everything())

final_spec <- boost_tree(
  trees = best_r2$trees,           # Number of boosting rounds (trees)
  tree_depth = best_r2$tree_depth, # Maximum depth of each tree
  min_n = best_r2$min_n,           # Minimum number of samples to split a node
  learn_rate = best_r2$learn_rate  # Learning rate (step size shrinkage)
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

final_spec

set.seed(10)
final_fit <- last_fit(final_spec,
                weather_recipe,
                split = weather_split)

final_fit %>%
  collect_predictions()

test_met <- final_fit %>%
  collect_metrics() %>% 
  mutate(estimate = round(.estimate, 3))

final_spec %>%
  fit(strength_gtex ~ .,
      data = bake(weather_prep, 
                  weather_train)) %>%
  augment(new_data = bake(weather_prep, 
                          weather_train)) %>% 
  rmse(strength_gtex, .pred) %>%
  bind_rows(
    
    
# R2
final_spec %>%
  fit(strength_gtex ~ .,
      data = bake(weather_prep, 
                  weather_train)) %>%
  augment(new_data = bake(weather_prep, 
                          weather_train)) %>% 
  rsq(strength_gtex, .pred))

final_fit %>%
  collect_predictions() %>%
  ggplot(aes(x = strength_gtex,
             y = .pred)) +
  geom_point() +
  geom_abline() +
  geom_smooth(method = "lm") +
  scale_x_continuous(limits = c(20, 40)) +
  scale_y_continuous(limits = c(20, 40)) 

publication_ready <- final_fit %>%
  collect_predictions() %>%
  ggplot(aes(x = strength_gtex,
             y = .pred)) +
  # Changing the shape of the points to 
  geom_point() +
  # Using colorblind friendly color scale 
  
  # Changing the color and linetype of abline
  geom_abline(color = "red", linetype = 2) +
  
  geom_smooth(method = "lm") +
  labs(x = "Observed Strenght (gtex)",
       y = "Predicted Strength (gtex)")+
  # Annotate R-sq and RMSE value in the plot
  
  
  scale_x_continuous(limits = c(20, 40)) +
  scale_y_continuous(limits = c(20, 40)) +
  
  # Changing the theme of the plot


cat(paste0("\nSaving...... publication ready plot for test dataset\n"))

ggsave(plot = publication_ready, 
       path = here("output", "png"),
       filename = "model_perf_test_data.png",
       height = 6,
       width = 9,
       dpi = 600)
  

vip <- final_spec %>%
  fit(strength_gtex ~ .,
         data = bake(weather_prep, weather_train)) %>% #There little change in variable improtance if you use full dataset
    vi() %>%
  mutate(
    Variable = fct_reorder(Variable, 
                           Importance)
  ) %>%
  ggplot(aes(x = Importance, 
             y = Variable,
             fill = Importance)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL)+
  theme(panel.background  = element_rect(fill = "gray82"),
        panel.grid = element_blank())

cat(paste0("\nSaving...... publication ready plot for variable importance\n"))

ggsave(plot = vip, 
       path = here("output/png"),
       filename = "vip_test_data.png",
       height = 6,
       width = 9,
       dpi = 600)

#knitr::purl("04-02_sapelo.qmd", output = "sapelo_script.R", documentation = 0)
