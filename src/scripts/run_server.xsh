from dotenv import dotenv_values

for key, value in dotenv_values('.env').items():
    ${key} = value

# $(python -u -m uvicorn wheat_mlops.api:app --host 0.0.0.0 --port 8080)
# $(python -u -m uvicorn src.python.wheat_mlops.api:app --host 0.0.0.0 --port 8080)
