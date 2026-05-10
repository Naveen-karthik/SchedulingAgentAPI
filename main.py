from fastapi import FastAPI
from fastapi.responses import JSONResponse
from service.agent_service import run_scheduling_agent

app = FastAPI(title="Scheduling Agent API")


@app.post("/api/schedule/run/")
def run_schedule():
    try:
        result = run_scheduling_agent()
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})