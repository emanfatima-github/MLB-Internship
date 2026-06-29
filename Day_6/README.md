# Day 6 — File Handling & JSON in Python

# What I Learned Today

Today I learned how to save that data to a file so it stays there even
after the program closes. This was a big moment because now my programs can
actually remember things.

I covered:
- How to create, read, write, and add to text files
- The 'with' statement — a safer way to open files
- File modes: 'r' to read, 'w' to write, 'a' to add without deleting
- JSON — a clean way to save structured data like a list of students
- How to convert Python data into JSON and load it back

# How File Handling and JSON Work Together

A text file is like a notebook — you can write anything in it, but it has
no structure. JSON is like a notebook with tables — everything has a label,
so Python can find exactly what it needs.

Together they let me:
1. Save a student's name, age, and grade in an organised way
2. Load that data back exactly as it was the next time I run the program
3. Change one student's grade without touching everyone else's data
