from fastapi import FastAPI
from app.routes.students import router


app = FastAPI(
    title="Student Management REST API"
)


app.include_router(router)
