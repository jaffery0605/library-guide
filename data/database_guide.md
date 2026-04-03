# Library Database Guide

This guide describes the database structure used by the Library Management System.

## Tables

### 1. Books
- `id`: Unique identifier for each book (Primary Key).
- `title`: Title of the book.
- `author`: Name of the author.
- `isbn`: International Standard Book Number.
- `genre`: Category (e.g., Fiction, Non-Fiction, Science).
- `available_copies`: Number of copies currently in the library.

### 2. Members
- `id`: Unique identifier for each member.
- `name`: Full name of the member.
- `email`: Contact email.
- `joined_date`: Date the member joined.

### 3. BorrowedBooks
- `id`: Transaction ID.
- `book_id`: Reference to the `Books` table.
- `member_id`: Reference to the `Members` table.
- `borrow_date`: Date the book was taken.
- `due_date`: Date the book should be returned.

## Data Types
- Most text fields use `VARCHAR(255)`.
- Dates are stored in `YYYY-MM-DD` format.
- IDs are `INTEGER` with `AUTO_INCREMENT`.
