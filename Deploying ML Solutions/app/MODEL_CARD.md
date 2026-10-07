# Model card: practice house-price estimator

Use this as a template for your own projects. Replace each section with facts
about your model, and update it whenever you retrain.

## Model details

- **Version:** 1.0.0 (see `model/metadata.json` for the training timestamp and library versions)
- **Type:** scikit-learn `Pipeline`: `StandardScaler`, then `KNeighborsRegressor(n_neighbors=10)`. A prediction is the average sale price of the 10 most similar training houses.
- **Owner:** KamiLimu Data Science track (teaching material)

## Intended use

- **Intended:** classroom practice in deploying, testing, and documenting a model; a tool for the help-desk agent example.
- **Users:** KamiLimu students and mentors.
- **Out of scope:** any real valuation, lending, insurance, tax, or purchase decision; any market other than the training data's; any period other than 2006-2010.

## Training data

- Ames, Iowa residential sales, 2006-2010 (`data/train.csv`, 1,460 rows; column definitions in `data/data_description.txt`).
- Six numeric features: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, `FullBath`.
- One small US city over five years. It says nothing about prices in Kenya or today.

## Evaluation

| Metric (random 80/20 validation split) | Value |
| --- | ---: |
| MAE, this model (scaled kNN) | about $20,900 |
| MAE, linear regression on the same features | about $25,300 |
| MAE, always predict the training median | about $59,600 |

A random split assumes future houses resemble past ones. A time-based split
would be a more honest test for forecasting future sales.

## Known limitations

- kNN cannot extrapolate: it can only average prices of houses it has seen. Inputs outside the training ranges return a warning, and their estimates are unreliable.
- Location is not a feature, although it strongly affects real prices.
- Errors are larger for very expensive houses, which are rare in the data.
- Feature effects are associations, not causes.

## Ethical considerations

- If location were added, it could act as a proxy for income or demographic group. A real valuation model would need a fairness review.
- The model receives no personal data; keep it that way.

## Monitoring

- Count range warnings per day (input drift).
- Track the distribution of predictions (prediction drift).
- When real prices are known, compare live MAE with the validation MAE above.
- Review user feedback logged by the agent.
