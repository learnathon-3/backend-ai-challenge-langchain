from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from src.graph_main import Agent

app = FastAPI()

@app.get("/run_quiz_bot")
def run_quiz_bot(message: str = Query(..., description="사용자가 전달하는 메시지")):
    try:
        agent = Agent()
        response_message = ""
        for response in agent.run(message):
            response_message += response

        return JSONResponse(
                status_code=202,
                content={
                    "message": response_message,
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

@app.get("/")
def root():
    return {"message": "AI Quizbot is running"}