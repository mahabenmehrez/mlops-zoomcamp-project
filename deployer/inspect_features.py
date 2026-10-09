import io
import pickle

import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

obj = s3.get_object(
    Bucket="mlops-zoomcamp-bucket",
    Key="models/skmodels/versions/v1/dict_vectorizer.pkl",
)
dv = pickle.loads(obj["Body"].read())

names = list(dv.get_feature_names_out())
print("Nombre de colonnes :", len(names))
print("5 premières :", names[:5])
print("5 dernières :", names[-5:])
print("Noms sans '=' (variables numériques) :", [n for n in names if "=" not in n])
print("Exemples avec 'PU' :", [n for n in names if "PU" in n][:5])