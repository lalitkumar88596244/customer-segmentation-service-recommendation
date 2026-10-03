# Customer Segmentation & Service Recommendation System
**Veda Technology | Major Project — Data Science Track**

## Objective
Group customers by behavior and engagement, inspect segment profiles, and provide example recommendations for services or learning programs.

## Files
- `app.py`: Streamlit dashboard, K-Means, PCA, profiles, recommendations.
- `customer_data.csv`: 300 synthetic records.
- `PROJECT_REPORT.pdf`: project report.
- `requirements.txt`: dependencies.

## Run
1. Install Python 3.10+.
2. Run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`.
4. Dashboard opens in your browser.

Features required for an uploaded CSV: `website_visits`, `engagement_score`, `monthly_spend`, `service_inquiries`, `support_requests`, `training_interest`. Customer IDs are optional.

## Method
Median imputation, feature standardization, K-Means clustering, PCA visualization, silhouette score, and transparent rule-based service suggestions.

## Data note
The included dataset is synthetic and is not Veda Technology's private customer data. Avoid uploading personal or confidential records. Interpret clusters only after reviewing profiles and validate recommendations before real use.
