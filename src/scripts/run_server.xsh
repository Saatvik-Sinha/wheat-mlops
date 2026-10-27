$MODEL_PATH = "data/working/trained/best.pt"
proc_ngrok = !(ngrok http --url=oxychloric-carma-subgelatinously.ngrok-free.dev 8080)
proc_server = !(python -u -m uvicorn wheat_mlops.api:app --host 0.0.0.0 --port 8080)
