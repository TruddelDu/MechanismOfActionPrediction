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
## Other Attempts
...coming soon
## Note to the Error function
... coming soon (What is log loss?
Why are the log loss in my scripts and on kaggle different?)
## Technologies
The project was created with:
- pandas 1.3.4
- sklearn 1.0.1
- xgboost 1.5.1
- logging 0.5.1.2
