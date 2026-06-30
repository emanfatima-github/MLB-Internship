# Day 7 – Object-Oriented Programming (OOP)

## What is Object-Oriented Programming?

Object-Oriented Programming (OOP) is a programming approach where we create programs using "classes" and "objects".
It helps organize code into smaller, reusable parts, making programs easier to understand, maintain, and update.

## Where Inheritance Was Used

In the Library Management System, I used inheritance by creating an "Item" parent class and a "Book" child class.
The 'Book' class inherits the 'title' attribute from the 'Item' class and also has its own attributes like 'author' and 'available'.
This helped reduce code duplication and made the code more organized.

## Challenges I Faced and How I Solved Them

One challenge I faced was understanding how to organize classes in Jupyter Notebook. I learned that the complete 'Library' class
should be written in a single code cell instead of splitting it across multiple cells.

I also handled invalid menu input using 'try' and 'except' so that the program does not crash when the user enters something other than a number.
