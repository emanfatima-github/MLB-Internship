from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Pydantic model
class Student(BaseModel):
    name: str
    age: int
    department: str


# Temporary student data
students = [
    {
        "id": 1,
        "name": "Ali",
        "age": 20,
        "department": "Computer Science"
    },
    {
        "id": 2,
        "name": "Sara",
        "age": 21,
        "department": "Software Engineering"
    }
]


# GET /
@app.get("/")
def home():
    return {"message": "Welcome to Student API"}


# GET /health
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# GET /students
@app.get("/students")
def get_students():
    return students


# POST /students
@app.post("/students")
def add_student(student: Student):
    new_id = len(students) + 1

    new_student = {
        "id": new_id,
        "name": student.name,
        "age": student.age,
        "department": student.department
    }

    students.append(new_student)

    return new_student


# GET /students/{id}
@app.get("/students/{id}")
def get_student(id: int):
    for student in students:
        if student["id"] == id:
            return student

    raise HTTPException(
        status_code=404,
        detail="Student not found"
    )
