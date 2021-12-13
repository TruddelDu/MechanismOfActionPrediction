# Mechanism of Action Prediction

This project concerns the Kaggle Competition "Mechanisms of Action (MoA) Prediction" (https://www.kaggle.com/c/lish-moa/overview).
This was my final project during my WBS Coding School Data Science Bootcamp.

In this competition the Mechanism of Action of various drugs (= what do the drugs do in the body) was to be predicted from *in vitro* data. 
I was able to predict the MoA with a log loss of 0.01900 in the private score (late submission). The winning team achieved a log loss of 0.01599.
## Final Pipeline
My final model uses all provided gene expression data as well as cell viability data. Treatment dose (label encoded) and duration was also used for the prediction. 
In contrast to many other competitors I did not consider the type of compound (actual compound vs control) for the prediction as it would lead to bleed through. 

As my model I used the XGBClassifier wrapped in sklearns OneVsRestClassifier with the following parameters:
- max_depth=6
- tree_method='hist' 
- scale_pos_weight=2
- min_child_weight=1
- max_delta_step=8
- eta=0.1
- objective='binary:logistic'
- eval_metric='logloss'
- subsample = 0.75
- use_label_encoder=False
## Other attempts
### Model selection
Besides XGBoostClassifier I also tried DesicionTreeClassifier, RandomForestClassifier, KNeighborsClassifier and TabnetClassifier.
Parameter tuning was only applied to XGBoostClassifier as it provided better inital predictions than both the DesicionTreeClassifier and KNeighborsClassifier. While the RandomForest model provided slightly better predictions it took 1.5-times as long to build. The TabnetClassifier model was to computationally heavy for my hardware.
### Feature engineering
Inspired from practices from microbiology I tested whether the fold-change of measurements (compared to the controls) instead of the absolute value would improve predictions. For this, I determined the mean of all features in controls per dose and incubation time. These showed a high variance. Then, I diveded each sample by the corresponding control-mean. This increased the summed log loss (see below) from 2.8 to 3.9 and was not persued further. 

Furthermore, I tested both Principal Component Analysis (PCA) and K-Best selection. PCA was performed seperately for genetic and cell viability data. Both methods decreased prediction quality. K-Best selected features did not impair prediction quality to badly and improved computation time drastically. However, the predictions made with K-Best selected features were unable to beat predictions made with all features.

So far, these new features were only used **instead** of the original features. As the fold-changes and PCA might hold more easily accessible information to the original features using all of these together might improve  predictions. 
## Note to the cost function
Log loss was used as the cost function in this project. Log loss is the negative average of the log of corrected predicted probabilities for each instance and is as such well suited for cost determination in classificatoin problems using probabilities.

To score my predictions of the internal test dataset I used sklearns log loss function. This function calculates the average log loss per sample summed over all features. Kaggle however uses a different log loss function to score the submissions. Here, the log loss is determined as the average per sample **and** feature. Both versions of the log loss function are proportional to each other, which means that sklearns summed log loss can be used for internal scoring even though the final score is determined slightly differently by kaggle.  
## Technologies
The project was created with:
- pandas 1.3.4
- sklearn 1.0.1
- xgboost 1.5.1
- logging 0.5.1.2
