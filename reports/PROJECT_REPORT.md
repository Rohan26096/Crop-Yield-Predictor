# Project Report
## Crop Yield Prediction Using Multi-Layer Perceptron Regression

> Complete the result tables after running `python train.py`. Do not insert invented metrics. The data section below refers to the cited Mendeley dataset record; verify the downloaded CSV and report the actual cleaning counts produced during your run.

## 1. Problem Definition and Literature Survey

### Problem definition
Agricultural yield varies with crop type, season, location, weather and farming inputs. Estimating crop yield from historical agricultural attributes can support planning and comparison. This project formulates crop-yield estimation as a supervised regression problem. A Multi-Layer Perceptron (MLP), a feed-forward artificial neural network, is trained to estimate the continuous yield value from crop, region, season, year, rainfall, cultivated area, fertilizer and pesticide attributes.

### Motivation
Farmers and agricultural planners need information about expected productivity for planning resources and assessing production patterns. A data-driven model can learn nonlinear relationships among historical features. This academic system demonstrates the workflow; it is not a certified agronomic advisory tool.

### Objectives
1. Obtain and inspect a real Indian crop-yield dataset.
2. Clean missing and invalid observations and preprocess numerical/categorical attributes.
3. Develop an MLP regression model and experiment with network parameters.
4. Evaluate predictions with MAE, MSE, RMSE and R².
5. Present dataset analytics and model predictions through a Flask dashboard.

### Literature and technical references
1. Ramesh, V. & Kumaresan, P. (2025). *Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques*. Mendeley Data, Version 2. DOI: 10.17632/ncw2vbcgnk.2. Dataset record describes Indian historical crop-yield observations and is the source selected for this project.
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). “Learning representations by back-propagating errors.” *Nature*, 323, 533–536. https://doi.org/10.1038/323533a0
3. Scikit-learn Developers. *MLPRegressor documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

## 2. Dataset and Preprocessing

### Dataset selection
The selected Mendeley Data record describes 19,689 crop observations across 27 Indian states and 3 Union Territories for 1997–2020, covering 55 crop types. The record lists crop, season, crop year, state, annual rainfall, area, production, pesticide, fertilizer and yield. The dataset record states CC BY 4.0. Cite the source in submitted work and retain its licence/attribution terms.

### Attributes
| Attribute | Role | Description |
|---|---|---|
| Crop | Input | Crop category |
| Season | Input | Growing season |
| Crop_Year | Input | Year of observation |
| State | Input | State/UT |
| Annual_Rainfall | Input | Annual rainfall |
| Area | Input | Cultivated area |
| Pesticide | Input | Pesticide quantity |
| Fertilizer | Input | Fertilizer quantity |
| Yield | Target | Continuous crop yield |
| Production | Excluded | Dropped to avoid target leakage |

**Target leakage note:** Yield is derived from production divided by area. Therefore, Production is not used as an input feature. The model is not allowed to see a direct ingredient of the target calculation.

### Preprocessing
- Column names are standardized to a common naming convention.
- Numeric columns are converted to numeric types; invalid values become missing.
- Rows without a target value are removed because supervised training requires a known target.
- Non-positive cultivated area and negative yield observations are removed as invalid for this modelling task.
- Numeric missing values are imputed using the median.
- Categorical missing values are imputed using the most frequent category.
- Crop, season and state are one-hot encoded. Unknown categories at prediction time are ignored by the encoder.
- Numeric predictors are standardized using StandardScaler.
- Preprocessing is placed inside a Scikit-learn Pipeline, so imputation, encoding and scaling are fitted only on training data and then applied consistently to validation/test data.
- Data is split into 80% training and 20% held-out test data (random_state=42). The training portion is further split for validation-based model selection.

## 3. ML Model Development

### MLP architecture
The MLP is a feed-forward artificial neural network:
- Input layer: transformed features from the preprocessing stage.
- Hidden layers: configurable dense layers, with candidate structures such as 32–16, 64–32 and 100–50 neurons.
- Hidden activation: ReLU or Tanh, depending on the experiment.
- Output layer: one linear output neuron for continuous yield prediction.

### Weight initialization
Scikit-learn's MLPRegressor initializes weights automatically using a random initialization scaled according to the layer sizes (Glorot-style bounds). The same random seed is used for reproducibility.

### Learning rate and optimizer
The Adam optimizer updates the network weights. Experiments use learning rates of 0.001 and 0.0005. The `alpha` parameter controls L2 regularization.

### Loss function
For regression, MLPRegressor minimizes squared-error loss (with its regularization term). The output is a continuous numeric yield estimate.

### Backpropagation and training
During forward propagation, inputs pass through the network to produce a yield estimate. The loss compares the estimate with the actual training target. Backpropagation applies the chain rule to calculate gradients of the loss with respect to weights and biases. Adam uses these gradients to update parameters. Training repeats over batches and epochs until the stopping condition is met. Early stopping monitors a validation fraction to reduce unnecessary training and help control overfitting.

## 4. Implementation and Experimentation

Implementation uses Python, Pandas, NumPy, Scikit-learn, Joblib and Flask. The training script tests five configurations. The best configuration is selected using validation MAE, then refitted on the full training partition. The held-out test set is used once for final performance reporting.

| Experiment | Hidden layers | Activation | Learning rate | L2 alpha |
|---|---|---|---:|---:|
| 1 | (32, 16) | ReLU | 0.001 | 0.0001 |
| 2 | (64, 32) | ReLU | 0.001 | 0.0001 |
| 3 | (100, 50) | ReLU | 0.001 | 0.0001 |
| 4 | (64, 32) | Tanh | 0.001 | 0.0001 |
| 5 | (64, 32) | ReLU | 0.0005 | 0.001 |

The actual experiment outcomes are saved to `reports/experiment_results.csv` after training.

## 5. Performance Evaluation

Regression metrics:
- **MAE:** average absolute difference between actual and predicted yield.
- **MSE:** average squared prediction error.
- **RMSE:** square root of MSE, expressed in the target's unit.
- **R²:** proportion of target variance explained by the model relative to a mean-baseline predictor. It can be negative when the model performs worse than that baseline.

Fill in from `models/metadata.json` after running training:

| Metric | Held-out test result |
|---|---:|
| MAE | Generated after training |
| MSE | Generated after training |
| RMSE | Generated after training |
| R² | Generated after training |

## 6. Analysis and Interpretation

Compare the five validation experiments using MAE and inspect test metrics for the selected model. Lower MAE and RMSE indicate smaller errors in the target unit. R² closer to 1 indicates stronger explanatory performance; negative R² indicates poor generalization. Discuss the actual values from the generated metadata and do not assume that the most complex network is best.

### Limitations
- Historical aggregated observations do not include every factor affecting yield, such as irrigation, soil properties, pest pressure and farm management.
- Random splitting can place observations from similar years/regions in both train and test sets. A time-based or region-based evaluation is a useful future extension.
- The model learns patterns in the source data and may not generalize to every farm or future climate condition.
- Prediction output is an estimate, not a guaranteed harvest result or agronomic recommendation.

### Conclusion
The project implements a complete supervised MLP regression workflow for crop-yield estimation: dataset preparation, feature preprocessing, neural-network training, parameter experiments, regression evaluation and a web-based prediction interface. The final conclusion must be written after examining the actual held-out test metrics and error patterns.

## 7. Demonstration and Viva

Demonstration steps:
1. Explain the crop-yield prediction problem and identify the target variable.
2. Show the source dataset and explain the input features.
3. Explain why Production was excluded to prevent target leakage.
4. Run the preprocessing and training script.
5. Show the five parameter experiments and selected configuration.
6. Explain hidden layers, activation, Adam, squared-error loss and backpropagation.
7. Show MAE, MSE, RMSE and R² from the test set.
8. Start Flask and enter a crop, state, season and numeric attributes.
9. Explain the predicted yield and its dataset unit.

### Sample viva questions
**1. Why is this a regression problem?**  
Because the output, crop yield, is a continuous numerical value.

**2. What is an MLP?**  
A feed-forward neural network containing an input layer, one or more hidden layers and an output layer.

**3. What is backpropagation?**  
It calculates gradients of the loss with respect to network weights and biases, propagating error information backward through the network.

**4. Why is feature scaling needed?**  
Features can have very different numeric ranges. Standardization helps the optimizer train more reliably.

**5. Why use one-hot encoding?**  
MLP models require numeric input. One-hot encoding represents categories without imposing an artificial numeric order.

**6. Why exclude Production?**  
Yield is computed from production and area. Including production would leak target information into the predictors.

**7. Why use MAE and RMSE?**  
Both measure prediction error in the target unit. RMSE penalizes larger errors more strongly.

**8. What does R² indicate?**  
It compares model prediction error with a baseline that predicts the mean target. Higher values are generally better.

**9. What is overfitting?**  
When a model learns training data too closely and performs poorly on unseen data.

**10. What is the role of the learning rate?**  
It controls the step size used when updating network weights during optimization.
