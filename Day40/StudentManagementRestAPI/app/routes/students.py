from fastapi import APIRouter, HTTPException
from app.schemas.student import Student


router = APIRouter()

students = []


@router.post("/students")
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


@router.get("/students")
def get_students():
    return students


@router.get("/students/{student_id}")
def get_student(student_id: int):
    for student in students:
        if student["id"] == student_id:
            return student

    raise HTTPException(
        status_code=404,
        detail="Student not found"
    )


@router.put("/students/{student_id}")
def update_student(student_id: int, student: Student):
    for existing_student in students:
        if existing_student["id"] == student_id:
            existing_student["name"] = student.name
            existing_student["age"] = student.age
            existing_student["department"] = student.department

            return existing_student

    raise HTTPException(
        status_code=404,
        detail="Student not found"
    )


@router.delete("/students/{student_id}")
def delete_student(student_id: int):
    for student in students:
        if student["id"] == student_id:
            students.remove(student)

            return {
                "message": "Student deleted successfully"
            }

    raise HTTPException(
        status_code=404,
        detail="Student not found"
    )
