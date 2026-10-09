import requests

from versioning import (
    BUCKET, LIVE, VERSIONS, FILES, s3,
    read_current, write_current,
)

API_URL = "http://localhost:8000"


def rollback():
    """Remet la dernière version approuvée par un humain dans best_model/,
    puis recharge l'API. Retourne la version restaurée."""
    state = read_current()
    target = state.get("last_approved")
    if not target:
        raise RuntimeError("Aucune version approuvée : rollback impossible.")

    for f in FILES:
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": f"{VERSIONS}/{target}/{f}"},
            Key=f"{LIVE}/{f}",
        )

    state["serving"] = target
    write_current(state)

    r = requests.post(f"{API_URL}/reload", timeout=30)
    r.raise_for_status()
    if r.json().get("result") != "success":
        raise RuntimeError(f"Reload refusé : {r.text}")
    return target


if __name__ == "__main__":
    print("Rollback vers :", rollback())