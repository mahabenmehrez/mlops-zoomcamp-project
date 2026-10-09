import json
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

BUCKET = "mlops-zoomcamp-bucket"
BASE = "models/skmodels"
LIVE = f"{BASE}/best_model"          # lu par l'API Flask
VERSIONS = f"{BASE}/versions"
CURRENT = f"{BASE}/current.json"
FILES = ["model.pkl", "dict_vectorizer.pkl"]

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)


def read_current():
    """Lit current.json, ou renvoie un état vide s'il n'existe pas."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=CURRENT)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"serving": None, "last_approved": None}
        raise


def write_current(state):
    s3.put_object(
        Bucket=BUCKET,
        Key=CURRENT,
        Body=json.dumps(state, indent=2).encode("utf-8"),
    )


def next_version_number():
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{VERSIONS}/", Delimiter="/")
    numbers = []
    for p in resp.get("CommonPrefixes", []):
        name = p["Prefix"].rstrip("/").split("/")[-1]   # ex: "v3"
        if name.startswith("v") and name[1:].isdigit():
            numbers.append(int(name[1:]))
    return max(numbers, default=0) + 1


def snapshot_current_model(note=""):
    """Copie best_model/ vers versions/vN/ et met à jour current.json."""
    n = next_version_number()
    version = f"v{n}"
    for f in FILES:
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": f"{LIVE}/{f}"},
            Key=f"{VERSIONS}/{version}/{f}",
        )
    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{VERSIONS}/{version}/metadata.json",
        Body=json.dumps(meta, indent=2).encode("utf-8"),
    )
    state = read_current()
    state["serving"] = version
    write_current(state)
    return version


def approve(version):
    """Marque une version comme approuvée par un humain."""
    state = read_current()
    state["last_approved"] = version
    write_current(state)


if __name__ == "__main__":
    print("État avant :", read_current())
    v = snapshot_current_model(note="baseline initiale (Gradient Boosting, RMSE 5.32)")
    print("Version créée :", v)
    print("État après :", read_current())