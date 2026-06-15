import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from itertools import combinations
import mlflow
import os
import json


if __name__ == "__main__":
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://[::1]:8081")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Research_Centre_v1")

    research_center = pd.read_csv('research_centers.csv')

    cols = ['internalFacilitiesCount', 'hospitals_10km', 'pharmacies_10km', 'facilityDiversity_10km', 'facilityDensity_10km']
    #cols = ['internalFacilitiesCount']
    
    # Get every combination from the cols list into a new list
    all_combinations = [list(c) for r in range(1, len(cols) + 1) for c in combinations(cols, r)]
    
    for cols in all_combinations:
        research_center_columns = research_center[cols]
        
        n_clusters, n_init, random_state = 3, 20, 42

        with mlflow.start_run():
            
            pipeline = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
            )
            
            pipeline.fit(research_center_columns)

            transformed_data = pipeline[:-1].transform(research_center_columns)
            labels = pipeline.named_steps["kmeans"].labels_
            score = silhouette_score(transformed_data, labels)
            
            print(f'Silhouette Score: {score}')
            
            mlflow.log_param('random_state', random_state)
            mlflow.log_param('n_init', n_init)
            mlflow.log_param('n_clusters', n_clusters)
            mlflow.log_metric('silhouette_score', score)
            
            # Get selected features
            selected_features = research_center_columns.columns[
                pipeline.named_steps["variancethreshold"].get_support()
            ].tolist()
            
            # Log as parameter instead (simpler, no file needed)
            mlflow.log_param('selected_features', str(selected_features))
            mlflow.log_param('n_features_selected', len(selected_features))
            
            # Log the model WITHOUT registered_model_name first to test
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path='model',
                registered_model_name="FEM"  # Try without this first
            )
            
            print(f"✅ Model trained and logged successfully")
            print(f"   - Silhouette Score: {score:.4f}")
            print(f"   - Features selected: {len(selected_features)}/{len(cols)}")
            print(f"   - Selected features: {selected_features}")