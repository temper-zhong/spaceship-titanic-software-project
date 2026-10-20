# Exploratory Data Analysis

## Data scope

- Training data: 8,693 rows and 14 columns.
- Test data: 4,277 rows and 13 columns.
- Positive target rate: 50.36%.
- Complete training rows: 6,606 (75.99%).
- Travel groups: 6,217; largest group: 8 passengers.

## Highest missing rates

| Feature | Train missing | Train rate | Test missing | Test rate |
|---|---:|---:|---:|---:|
| CryoSleep | 217 | 2.50% | 93 | 2.17% |
| ShoppingMall | 208 | 2.39% | 98 | 2.29% |
| VIP | 203 | 2.34% | 93 | 2.17% |
| HomePlanet | 201 | 2.31% | 87 | 2.03% |
| Name | 200 | 2.30% | 94 | 2.20% |

Missingness is spread across every predictor except `PassengerId`. Rates are low and similar between train and test, so there is no obvious train/test missingness shift.

## Data-quality and anomaly checks

| Check | Count | Interpretation |
|---|---:|---|
| duplicate_train_rows | 0 | expected 0 |
| duplicate_test_rows | 0 | expected 0 |
| duplicate_passenger_ids | 0 | expected 0 |
| invalid_passenger_id_format | 0 | expected 0 |
| invalid_cabin_format | 0 | expected 0 |
| negative_age | 0 | expected 0 |
| age_above_100 | 0 | review if nonzero |
| negative_spend_values | 0 | expected 0 |
| cryo_sleep_with_positive_spend | 0 | expected 0 under domain constraint |

IQR outlier counts are included in `outputs/data_quality.csv`. High spending values are retained because they are plausible passenger charges and tree models can use them without arbitrary clipping.

## Main findings

1. Cryogenic sleep is the strongest raw segment: transported rates are 81.8% for sleeping passengers and 32.9% for awake passengers.
2. Spending separates outcomes: zero-spend passengers have a 78.6% transport rate versus 29.9% for passengers with positive recorded spending.
3. Children aged 12 or below have a 70.0% transport rate versus 48.3% for older passengers.
4. Cabin side matters: side S has a 55.5% rate versus 45.1% for side P.
5. Group travel matters: solo passengers have a 45.2% rate versus 56.7% for passengers in multi-person groups.
6. Deck-level rates are heterogeneous: B=73.4%, C=68.0%, G=51.6%, A=49.6%, F=44.0%, D=43.3%, E=35.7%, T=20.0%.

These findings motivate the cryosleep, spending, cabin, age, and travel-party features used by the modelling pipeline.

## Figures

- `reports/figures/missingness.png`
- `reports/figures/target_distribution.png`
- `reports/figures/categorical_transport_rates.png`
- `reports/figures/numeric_distributions.png`
- `reports/figures/group_size_transport_rate.png`
- `reports/figures/correlation_heatmap.png`
