import os
from datetime import datetime

import streamlit as st

from src.dataframe import CSVDataFrame

# Path to main dataset
DATA_PATH = "data/goodreads_books.csv"

# Columns to show in most previews
PREVIEW_COLS = [
    "bookID",
    "title",
    "authors",
    "average_rating",
    "num_pages",
    "ratings_count",
    "text_reviews_count",
    "publication_date",
    "publisher",
]


# ----------------- Helpers ----------------- #
def show_table(rows, height=400):
    """
    Display a list of dict rows as a scrollable table with readable headers.
    """
    st.dataframe(rows, use_container_width=True, height=height)

@st.cache_data
def load_books(path: str) -> CSVDataFrame:
    """Load the Goodreads dataset using our custom CSVDataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    return CSVDataFrame(path, separator=",")


def get_numeric_columns(df_obj: CSVDataFrame):
    """Detect numeric columns (int or float)."""
    numeric_cols = []
    for col, values in df_obj.data.items():
        for v in values:
            if isinstance(v, (int, float)) and v is not None:
                numeric_cols.append(col)
                break
    return numeric_cols


def get_row_count(df_obj: CSVDataFrame) -> int:
    """Number of rows in the dataframe."""
    if not df_obj.columns:
        return 0
    return len(df_obj.data[df_obj.columns[0]])


def build_rows(df_obj: CSVDataFrame, cols=None, indices=None):
    """Return list of dicts representing rows for given columns and indices."""
    if cols is None:
        cols = df_obj.columns
    if indices is None:
        indices = range(get_row_count(df_obj))

    rows = []
    for i in indices:
        row = {}
        for c in cols:
            row[c] = df_obj.data[c][i]
        rows.append(row)
    return rows


def sorted_indices(df_obj: CSVDataFrame, sort_col: str, ascending: bool = False):
    """Return row indices sorted by a column (None values at the end)."""
    col_vals = df_obj.data[sort_col]

    def key_func(i):
        v = col_vals[i]
        return (v is None, v)

    idxs = list(range(len(col_vals)))
    idxs.sort(key=key_func, reverse=not ascending)
    return idxs


@st.cache_data
def build_author_stats(_df) -> CSVDataFrame:
    author_mean = _df.groupby_mean("authors", "average_rating")
    author_count = _df.groupby_count("authors", "average_rating")

    author_stats = {
        "authors": author_mean["authors"],
        "author_avg_rating": author_mean["average_rating"],
        "author_book_count": author_count["average_rating"],
    }
    return CSVDataFrame.from_data(author_stats)


# ----------------- App ----------------- #

def main():
    st.title("Goodreads Books Analytics App")
    st.caption("Built on a custom CSV parser and DataFrame")

    # Load data
    try:
        df = load_books(DATA_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    numeric_cols = get_numeric_columns(df)
    author_df = build_author_stats(df)

    tabs = st.tabs(
        [
            "Overview",
            "Filter",
            "Projection",
            "Groupby & Aggregation",
            "Author Analytics (Join)",
        ]
    )

    # -------- Overview -------- #
    with tabs[0]:
        st.header("Dataset Overview")

        st.write("**Columns:**")
        st.write(df.columns)

        first_col = df.columns[0]
        total_rows = len(df[first_col])
        st.write(f"**Total books:** {total_rows}")

        st.subheader("Preview (first 20 rows)")
        preview_cols = [c for c in PREVIEW_COLS if c in df.columns]
        preview_rows = build_rows(
            df,
            cols=preview_cols,
            indices=range(min(20, total_rows)),
        )
        show_table(preview_rows)

        st.subheader("Quick global stats (using aggregate functions)")

        # Mean rating
        if "average_rating" in df.columns:
            st.write(f"Mean rating: {df.aggregate_mean('average_rating'):.3f}")

        # Max pages with book title
        if "num_pages" in df.columns and "title" in df.columns:
            max_pages = df.aggregate_max("num_pages")
            try:
                idx = df["num_pages"].index(max_pages)
                title = df["title"][idx]
                st.write(f"Max pages: {max_pages} ({title})")
            except ValueError:
                st.write(f"Max pages: {max_pages}")

        # Book with max number of ratings
        if "ratings_count" in df.columns and "title" in df.columns:
            max_ratings = df.aggregate_max("ratings_count")
            try:
                idx = df["ratings_count"].index(max_ratings)
                title = df["title"][idx]
                st.write(
                    f"Book with max number of ratings: {title} ({max_ratings} ratings)"
                )
            except ValueError:
                st.write(f"Max ratings_count: {max_ratings}")

        # Book with max number of reviews
        if "text_reviews_count" in df.columns and "title" in df.columns:
            max_reviews = df.aggregate_max("text_reviews_count")
            try:
                idx = df["text_reviews_count"].index(max_reviews)
                title = df["title"][idx]
                st.write(
                    f"Book with max number of reviews: {title} ({max_reviews} reviews)"
                )
            except ValueError:
                st.write(f"Max text_reviews_count: {max_reviews}")

    # -------- Filter -------- #
    with tabs[1]:
        st.header("Filter Books")

        st.write(
            "This tab demonstrates the `filter` function. "
            "You can filter on one numeric column at a time."
        )

        # Exclude bookID and isbn13 from numeric filters
        numeric_for_filter = [c for c in numeric_cols if c not in ("bookID", "isbn13")]

        if not numeric_for_filter:
            st.info("No suitable numeric columns available for filtering.")
        else:
            col = st.selectbox("Choose numeric column", numeric_for_filter)

            col_values = [v for v in df[col] if isinstance(v, (int, float))]
            if col_values:
                min_val = min(col_values)
                max_val = max(col_values)

                op = st.selectbox(
                    "Condition", [">=", "<=", ">", "<", "==", "between"]
                )

                is_int_col = all(
                    (v is None or isinstance(v, int)) for v in df[col]
                )

                if op == "between":
                    if is_int_col:
                        low, high = st.slider(
                            "Range",
                            int(min_val),
                            int(max_val),
                            (int(min_val), int(max_val)),
                        )
                    else:
                        low, high = st.slider(
                            "Range",
                            float(min_val),
                            float(max_val),
                            (float(min_val), float(max_val)),
                        )

                    def cond(x, lo=low, hi=high):
                        return x is not None and lo <= x <= hi

                else:
                    if is_int_col:
                        slider_thr = st.slider(
                            "Threshold",
                            int(min_val),
                            int(max_val),
                            int(min_val),
                        )
                        manual_thr = st.number_input(
                            "Threshold (manual input)",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(slider_thr),
                        )
                        thr = manual_thr
                    else:
                        slider_thr = st.slider(
                            "Threshold",
                            float(min_val),
                            float(max_val),
                            float(min_val),
                        )
                        manual_thr = st.number_input(
                            "Threshold (manual input)",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(slider_thr),
                        )
                        thr = manual_thr

                    if op == ">=":
                        def cond(x, t=thr):
                            return x is not None and x >= t
                    elif op == "<=":
                        def cond(x, t=thr):
                            return x is not None and x <= t
                    elif op == ">":
                        def cond(x, t=thr):
                            return x is not None and x > t
                    elif op == "<":
                        def cond(x, t=thr):
                            return x is not None and x < t
                    else:  # "=="
                        def cond(x, t=thr):
                            return x is not None and x == t

                if st.button("Apply numeric filter", type="primary"):
                    filtered_df = df.filter(col, cond)
                    count = get_row_count(filtered_df)
                    st.success(f"Filtered rows: {count}")

                    # Limit rows for display
                    max_show = 100
                    preview_cols = [c for c in PREVIEW_COLS if c in filtered_df.columns]
                    indices = range(min(max_show, count))
                    head_rows = build_rows(
                        filtered_df,
                        cols=preview_cols,
                        indices=indices,
                    )
                    show_table(head_rows)

    # -------- Projection -------- #
    with tabs[2]:
        st.header("Projection (Select Columns)")

        st.write(
            "This tab demonstrates the `select` function: "
            "you can choose a subset of columns to view."
        )

        default_cols = [
            c
            for c in ["bookID", "title", "authors", "average_rating"]
            if c in df.columns
        ]
        selected_cols = st.multiselect(
            "Columns to keep", df.columns, default=default_cols
        )

        if selected_cols:
            projected = df.select(selected_cols)
            count = get_row_count(projected)
            st.write(f"Projected columns: {projected.columns}")

            max_show = 50
            indices = range(min(max_show, count))
            head_rows = build_rows(
                projected,
                cols=selected_cols,
                indices=indices,
            )
            show_table(head_rows)
        else:
            st.info("Select at least one column to view a projection.")

    # -------- Groupby & Aggregation -------- #
    with tabs[3]:
        st.header("Groupby & Aggregation")

        st.write(
            "Here we use the `groupby_*` and `aggregate_*` functions to "
            "compute predefined, meaningful summaries of the dataset."
        )

        # 1) Average rating by language_code
        st.subheader("Average rating by language_code")
        if "language_code" in df.columns and "average_rating" in df.columns:
            lang_group = df.groupby_mean("language_code", "average_rating")
            idxs = sorted_indices(lang_group, "average_rating", ascending=False)
            rows = build_rows(
                lang_group,
                cols=["language_code", "average_rating"],
                indices=idxs[:20],
            )
            show_table(rows)
        else:
            st.info("language_code or average_rating not available.")

        # 2) Top publishers by number of books
        st.subheader("Top publishers by number of books")
        if "publisher" in df.columns and "bookID" in df.columns:
            pub_group = df.groupby_count("publisher", "bookID")
            # Rename bookID -> book_count
            pub_group.data["book_count"] = pub_group.data["bookID"]
            pub_group.columns = ["publisher", "book_count"]
            del pub_group.data["bookID"]

            idxs = sorted_indices(pub_group, "book_count", ascending=False)
            rows = build_rows(
                pub_group,
                cols=["publisher", "book_count"],
                indices=idxs[:20],
            )
            show_table(rows)
        else:
            st.info("publisher or bookID not available.")

        # 3) Authors with at least 5 books
        st.subheader("Authors with at least 5 books (sorted by avg rating)")
        if "authors" in df.columns and "average_rating" in df.columns:
            # author_df is already built and cached
            filtered_authors = author_df.filter(
                "author_book_count",
                lambda x: x is not None and x >= 5,
            )
            if get_row_count(filtered_authors) == 0:
                st.info("No authors with at least 5 books.")
            else:
                idxs = sorted_indices(
                    filtered_authors, "author_avg_rating", ascending=False
                )
                rows = build_rows(
                    filtered_authors,
                    cols=["authors", "author_avg_rating", "author_book_count"],
                    indices=idxs[:20],
                )
                show_table(rows)
        else:
            st.info("authors or average_rating not available.")

        # 4) Global aggregates
        st.subheader("Global aggregates")
        if "average_rating" in df.columns:
            st.write(f"Overall mean rating: {df.aggregate_mean('average_rating'):.3f}")
        if "num_pages" in df.columns:
            st.write(f"Overall mean pages: {df.aggregate_mean('num_pages'):.1f}")
            st.write(f"Max pages: {df.aggregate_max('num_pages')}")

    # -------- Author Analytics (Join) -------- #
    with tabs[4]:
        st.header("Author Analytics (using Join)")

        st.write(
            "We compute author-level statistics using groupby, filter on them, "
            "and join back to the books table using `inner_join`."
        )

        if "authors" not in df.columns or "average_rating" not in df.columns:
            st.error(
                "Dataset does not have required columns `authors` and `average_rating`."
            )
        else:
            # Filter authors with many books and high avg rating (static conditions)
            filtered_authors = author_df.filter(
                "author_book_count",
                lambda x: x is not None and x >= 5,
            )
            filtered_authors = filtered_authors.filter(
                "author_avg_rating",
                lambda x: x is not None and x >= 4.2,
            )

            if get_row_count(filtered_authors) == 0:
                st.info("No authors meet the chosen static conditions.")
            else:
                st.subheader("Top authors (book_count >= 5, avg_rating >= 4.2)")

                # Show top 20 authors by avg rating
                idxs = sorted_indices(
                    filtered_authors, "author_avg_rating", ascending=False
                )
                author_rows = build_rows(
                    filtered_authors,
                    cols=["authors", "author_avg_rating", "author_book_count"],
                    indices=idxs[:20],
                )
                show_table(author_rows)

                # Join books with these authors
                joined = df.inner_join(filtered_authors, on="authors")

                # We will show books for top 5 authors, grouped in expanders
                st.subheader("Books by top authors (joined view)")

                joined_authors = joined["authors"]
                cols_to_show = [
                    c
                    for c in [
                        "bookID",
                        "title",
                        "authors",
                        "average_rating",
                        "num_pages",
                        "ratings_count",
                        "author_avg_rating",
                        "author_book_count",
                    ]
                    if c in joined.columns
                ]

                # Top 5 authors from filtered_authors
                top_k = 5
                top_indices = idxs[:top_k]
                top_authors = [filtered_authors["authors"][i] for i in top_indices]
                top_counts = [
                    filtered_authors["author_book_count"][i] for i in top_indices
                ]
                top_avgs = [
                    filtered_authors["author_avg_rating"][i] for i in top_indices
                ]

                for k, author_name in enumerate(top_authors):
                    # Find rows in joined where authors == author_name
                    j_indices = [
                        i for i, a in enumerate(joined_authors) if a == author_name
                    ]
                    if not j_indices:
                        continue

                    book_count = top_counts[k]
                    avg_r = top_avgs[k]
                    label = (
                        f"{author_name} ({book_count} books, "
                        f"avg rating {avg_r:.2f})"
                    )

                    with st.expander(label):
                        # Limit to first 10 books for display
                        show_indices = j_indices[:10]
                        rows = build_rows(joined, cols=cols_to_show, indices=show_indices)
                        show_table(rows)


if __name__ == "__main__":
    main()