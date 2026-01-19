
# Methods

## Clinical Context
FSGS is a progressive kidney disorder characterized by scarring in the glomeruli. This study focuses on identifying clinical predictors of survival and disease progression.

## Data Preprocessing
- Missing values imputed using median strategy
- Categorical variables encoded via one-hot encoding
- Features normalized using standard scaling

## Survival Modeling
- Cox Proportional Hazards model fitted using lifelines
- Survival time and event occurrence defined per patient record
- Risk scores computed and stratified into tertiles

## Evaluation
- Concordance index used to assess model performance
- Kaplan-Meier curves plotted for risk groups
- Hazard ratios visualized for interpretability
