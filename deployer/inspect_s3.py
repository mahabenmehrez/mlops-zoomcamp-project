import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

BUCKET = "mlops-zoomcamp-bucket"
PREFIX = "models/skmodels/"

resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
for obj in resp.get("Contents", []):
    print(obj["Key"], obj["Size"], obj["LastModified"])