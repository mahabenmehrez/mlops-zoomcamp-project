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
results = []
for z in sorted(set(zones)):
    m = zones == z
    if m.sum() < MIN_TRIPS:
        continue
    _, p = mannwhitneyu(err_new[m], err_old[m], alternative="greater")
    results.append((z, int(m.sum()), err_old[m].mean(), err_new[m].mean(), p))

tested = len(results)
alpha_corr = ALPHA / tested if tested else ALPHA
strict = [r for r in results if r[4] < ALPHA]
corrected = [r for r in results if r[4] < alpha_corr]

print(f"\nZones testées : {tested} (au moins {MIN_TRIPS} trajets chacune)")
print(f"Seuil strict = {ALPHA} | seuil corrigé (Bonferroni) = {alpha_corr:.4f}")
print(f"Zones en régression (règle stricte)  : {len(strict)}")
for z, n, a, b, p in strict:
    print(f"  {z}: n={n} MAE {a:.2f} -> {b:.2f} (p={p:.4f})")
print(f"Zones en régression (règle corrigée) : {len(corrected)}")

block_strict = p_global < ALPHA or len(strict) > 0
block_corr = p_global < ALPHA or len(corrected) > 0
print("\nDECISION stricte  :", "BLOQUER" if block_strict else "autoriser (niveau 3 : demander à un humain)")
print("DECISION corrigée :", "BLOQUER" if block_corr else "autoriser (niveau 3 : demander à un humain)")