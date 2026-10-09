import io
import os
import pickle

import boto3
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import mannwhitneyu

BUCKET = os.getenv("S3_BUCKET_NAME", "mlops-zoomcamp-bucket")
TEST_KEY = "nyc-taxi-data/datasets/year=2026/month=10/test.parquet"
DV_KEY = "models/skmodels/versions/v1/dict_vectorizer.pkl"
ALPHA = 0.05
MIN_TRIPS = 20  # en dessous, le test n'est pas fiable

mlflow.set_tracking_uri("http://mlflow:5000")
s3 = boto3.client(
    "s3",
    endpoint_url="http://localstack:4566",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    region_name="us-east-1",
)


def load_model(run):
    uri = f"{run.info.artifact_uri}/{run.data.params['model_name']}"
    return mlflow.pyfunc.load_model(uri)


client = MlflowClient()
exp = client.get_experiment_by_name("mlops-zoomcamp-experiment")
runs = client.search_runs(
    [exp.experiment_id], order_by=["metrics.rmse ASC"]
)
old_run, new_run = runs[0], runs[1]   # ancien = meilleur RMSE, candidat = 2e
print("ANCIEN   :", old_run.data.params["model_name"], old_run.data.metrics["rmse"])
print("CANDIDAT :", new_run.data.params["model_name"], new_run.data.metrics["rmse"])

# données de test et noms de colonnes
obj = s3.get_object(Bucket=BUCKET, Key=TEST_KEY)
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
X, y = df.drop(columns="target"), df["target"].to_numpy()

dv = pickle.loads(s3.get_object(Bucket=BUCKET, Key=DV_KEY)["Body"].read())
names = list(dv.get_feature_names_out())
pu_idx = [i for i, n in enumerate(names) if n.startswith("PULocationID=")]
pu_block = X.iloc[:, pu_idx].to_numpy()
zones = np.where(
    pu_block.sum(axis=1) > 0,
    np.array([names[pu_idx[j]] for j in pu_block.argmax(axis=1)]),
    "inconnue",
)

# erreurs absolues par trajet
err_old = np.abs(y - np.asarray(load_model(old_run).predict(X)).ravel())
err_new = np.abs(y - np.asarray(load_model(new_run).predict(X)).ravel())

print(f"\nMAE globale  ancien={err_old.mean():.3f}  candidat={err_new.mean():.3f}")
_, p_global = mannwhitneyu(err_new, err_old, alternative="greater")
print(f"Test global (candidat pire ?) p={p_global:.4f}")

# test par zone : H1 = les erreurs du candidat sont plus grandes
regressions, tested = [], 0
for z in sorted(set(zones)):
    m = zones == z
    if m.sum() < MIN_TRIPS:
        continue
    tested += 1
    _, p = mannwhitneyu(err_new[m], err_old[m], alternative="greater")
    if p < ALPHA:
        regressions.append((z, int(m.sum()), err_old[m].mean(), err_new[m].mean(), p))

print(f"\nZones testées : {tested} (au moins {MIN_TRIPS} trajets chacune)")
print(f"Zones en régression : {len(regressions)}")
for z, n, a, b, p in regressions:
    print(f"  {z}: n={n} MAE {a:.2f} -> {b:.2f} (p={p:.4f})")

blocked = p_global < ALPHA or len(regressions) > 0
print("\nDECISION :", "BLOQUER la promotion" if blocked else "Promotion autorisée (niveau 3 : demander à un humain)")