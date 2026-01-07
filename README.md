# Goodreads Books Analytics App

## Custom CSV Parser, DataFrame Implementation, and Streamlit Web Application

## Overview

This project implements a custom CSV parsing system and an in-memory DataFrame engine from scratch without using any data-processing libraries such as pandas, csv, or json.
It also includes a Streamlit web application that allows interactive data exploration on a real dataset of Goodreads books.

The project demonstrates:
	•	Custom CSV parsing
	•	In-memory column-oriented data storage
	•	SQL-like operations implemented manually
	•	A web application built on top of these functions


## Project Structure

```
project/
│
├── app.py                     # Streamlit web application
├── README.md                  # Documentation
│
├── src/
│   └── dataframe.py           # Custom CSV parser and DataFrame implementation
│
└── data/
    └── goodreads_books.csv    # Dataset used by the application
```

## Installation

### 1.	(optional) Create and activate an environment

### 2.	Install Streamlit
```
    pip install streamlit
```
No additional libraries are required.

## Running the Application

From the project root directory:
```
    streamlit run app.py
```
The local URL (usually http://localhost:8501) will be displayed in the terminal.

## Features Demonstrated

### 1. CSV Parsing

The CSVDataFrame class includes:
	•	Manual line-by-line file reading
	•	Quote handling and escaped quote handling
	•	Custom splitting logic
	•	Header cleaning
	•	Type inference for int, float, and string
	•	Error handling for malformed rows

This serves as a manual replacement for pandas.read_csv().

### 2. DataFrame Capabilities

The DataFrame implementation supports:
	•	Column access using df["column"]
	•	Filtering with custom conditions
	•	Projection (column selection)
	•	Group By operations
	•	Aggregations (sum, mean, min, max, std, count)
	•	Inner, left, right, and outer joins
	•	Convenience wrapper functions for common operations

### 3. Streamlit Web Application

The web app provides an interactive interface that uses all the custom functions above. It includes:

Dataset Overview
	•	Summary statistics
	•	Top rows displayed in a scrollable table

Filtering
	•	Numeric filters
	•	Manual threshold input
	•	Categorical filtering
	•	Date filtering
	•	Combined conditions

Projection
	•	Select any subset of columns for display

Group By and Aggregation
The app shows several predefined analytics, such as:
	•	Average rating per language
	•	Publisher averages
	•	Overall rating and page statistics

Join Usage
Author-level statistics are computed using groupby operations:
	•	Average rating per author
	•	Number of books per author

These statistics form a second table which is then joined back to the original dataset using:
```
    authors  →  avg_rating, book_count
```
Joins displayed in the app use the custom join logic implemented in dataframe.py.

## Dataset

The application uses the Goodreads Books dataset (approximately 11k rows), containing:
	•	bookID
	•	title
	•	authors
	•	average_rating
	•	isbn, isbn13
	•	language_code
	•	num_pages
	•	ratings_count
	•	text_reviews_count
	•	publication_date
	•	publisher

This dataset is large enough to demonstrate meaningful analytics while still manageable with a custom parser.

## Team

Single-person project: Ashvanth Rathinavel.
All implementation, testing, and development completed individually.
