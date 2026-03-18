import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import joblib
color_palette = ["#440154", "#482677", "#404788", "#33638d", "#287d8e",
"#1f968b", '#29af7f', '#55c667', '#73d055', '#b8de29', '#fde725']
fp = matplotlib.font_manager.FontProperties(fname='/Fonts/roboto/Roboto-Condensed.ttf')
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# import the data 
research_center = pd.read_csv('research_centers.csv')

# Get the numeric columns
cols = ['internalFacilitiesCount', 'hospitals_10km', 'pharmacies_10km', 'facilityDiversity_10km', 'facilityDensity_10km']

research_center_columns = research_center[cols]

pipeline = make_pipeline(
    StandardScaler(),
    VarianceThreshold(),
    KMeans(n_clusters=3, n_init=20, random_state=42)
)

pipeline.fit(research_center_columns)

score = silhouette_score(pipeline[:-1].transform(research_center_columns),
                         pipeline.named_steps["kmeans"].labels_)
print(f'Silhoutte Score is : {score}' )

# Save this in artifact folder
os.makedirs("model/artifacts", exist_ok=True)

joblib.dump({
    "pipeline": pipeline,
    "selected_features": research_center_columns.columns[pipeline.named_steps["variancethreshold"].get_support()].tolist()
}, "model/artifacts/kmeans_pipeline_model.pkl")

print("trained")