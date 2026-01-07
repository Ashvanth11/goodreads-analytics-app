class CSVDataFrame:
    def __init__(self, filename, separator=","):
        self.columns = []
        self.data = {}
        self._load_csv(filename, separator)
    
    def _load_csv(self, filename, separator):
        with open(filename, 'r', encoding='utf-8') as file:
            header_line = file.readline().rstrip('\n')
            if not header_line:
                raise ValueError("CSV file is empty or missing header")

            # Clean header names
            self.columns = [
                col.strip().replace('"', '').replace("'", '').replace('\ufeff', '')
                for col in header_line.split(separator)
            ]
            for col in self.columns:
                self.data[col] = []

            for line in file:
                line = line.rstrip('\n')
                if not line:
                    continue

                fields = self._parse_line(line, separator)

            
                # Schema (typical): bookID, title, authors, average_rating, isbn, isbn13, language_code, ...
                # So 'authors' is column index 2.
                if len(fields) == len(self.columns) + 1:
                    # Merge the split author fields back together
                    # fields[2] = 'Sam Bass Warner' , fields[3] = ' Jr./Sam B. Warner'
                    fields[2] = fields[2] + separator + fields[3]
                    del fields[3]

                # After attempting the fix, verify field count
                if len(fields) != len(self.columns):
                    raise ValueError(f"Malformed line (field count mismatch): {line}")

                for col, value in zip(self.columns, fields):
                    self.data[col].append(value)

        # Convert column types after reading all data
        self._infer_and_convert_types()
    
    def _parse_line(self, line, separator):
        values = []
        current_value = ""
        in_quotes = False
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"':
                if in_quotes and i+1 < len(line) and line[i+1] == '"':
                    # Escaped quote
                    current_value += '"'
                    i += 1
                else:
                    in_quotes = not in_quotes
            elif char == separator and not in_quotes:
                values.append(current_value)
                current_value = ""
            else:
                current_value += char
            i += 1
        values.append(current_value)
        return values
    
    def _infer_and_convert_types(self):
        for col, values in self.data.items():
            all_int = True
            all_float = True
            for v in values:
                if v == "" or v is None:
                    continue
                try:
                    int(v)
                except:
                    all_int = False
                try:
                    float(v)
                except:
                    all_float = False
            # Determine final type
            if all_int and not all_float:
                col_type = int
            elif all_float and not all_int:
                col_type = float
            elif all_int and all_float:
                col_type = float 
            else:
                col_type = str
            # Convert values to that type
            if col_type in (int, float):
                new_list = []
                for v in values:
                    if v == "" or v is None:
                        new_list.append(None)
                    else:
                        new_list.append(col_type(v))
                self.data[col] = new_list
        # (If col_type is str, we leave the list as-is)
    
    def __getitem__(self, col_name):
        if col_name in self.data:
            return self.data[col_name]
        else:
            raise KeyError(f"Column '{col_name}' not found")
        
    def filter(self, column, condition):
        """Return rows where condition(value) is True."""
        indices = [i for i, val in enumerate(self.data[column]) if condition(val)]
        new_data = {col: [self.data[col][i] for i in indices] for col in self.columns}
        return CSVDataFrame.from_data(new_data)

    def select(self, columns):
        """Return a new DataFrame with selected columns."""
        new_data = {col: self.data[col] for col in columns if col in self.data}
        return CSVDataFrame.from_data(new_data)

    @classmethod
    def from_data(cls, data_dict):
        """Create a DataFrame directly from a dict (used internally)."""
        obj = cls.__new__(cls)
        obj.columns = list(data_dict.keys())
        obj.data = data_dict
        return obj

    def groupby(self, group_col, agg_col, agg_func):
        """Group by values in 'group_col' and aggregate 'agg_col' using agg_func."""
        groups = {}

        for gval, aval in zip(self.data[group_col], self.data[agg_col]):
            if gval not in groups:
                groups[gval] = []
            groups[gval].append(aval)

        # Apply the aggregation function to each group
        result_data = {group_col: [], agg_col: []}
        for gval, values in groups.items():
            # Handle None or empty values gracefully
            clean_values = [v for v in values if v is not None]
            if clean_values:
                result_data[group_col].append(gval)
                result_data[agg_col].append(agg_func(clean_values))

        return CSVDataFrame.from_data(result_data)
    
    # --- Convenience wrappers for common groupby aggregations ---

    def groupby_sum(self, group_col, agg_col):
        """Group by a column and compute the sum of another column."""
        return self.groupby(group_col, agg_col, sum)

    def groupby_max(self, group_col, agg_col):
        """Group by a column and compute the maximum of another column."""
        return self.groupby(group_col, agg_col, max)

    def groupby_min(self, group_col, agg_col):
        """Group by a column and compute the minimum of another column."""
        return self.groupby(group_col, agg_col, min)

    def groupby_count(self, group_col, agg_col):
        """Group by a column and count the number of records in each group."""
        return self.groupby(group_col, agg_col, len)

    def groupby_mean(self, group_col, agg_col):
        """Group by a column and compute the mean of another column."""
        return self.groupby(group_col, agg_col, lambda x: sum(x)/len(x))

    def groupby_std(self, group_col, agg_col):
        """Group by a column and compute the standard deviation of another column."""
        import math
        def std_func(x):
            mean = sum(x)/len(x)
            return math.sqrt(sum((v - mean)**2 for v in x) / len(x))
        return self.groupby(group_col, agg_col, std_func)

    def aggregate(self, agg_col, agg_func):
        """Apply an aggregation function to a single column (no group by)."""
        if agg_col not in self.data:
            raise KeyError(f"Column '{agg_col}' not found")

        # Extract non-null values
        values = [v for v in self.data[agg_col] if v is not None]

        if not values:
            return None  # handle empty or all-null columns gracefully

        # Apply the aggregation function
        return agg_func(values)
    
    # --- Convenience wrappers for aggregate (no group by) ---

    def aggregate_sum(self, agg_col):
        """Return the sum of a column."""
        return self.aggregate(agg_col, sum)

    def aggregate_max(self, agg_col):
        """Return the maximum value of a column."""
        return self.aggregate(agg_col, max)

    def aggregate_min(self, agg_col):
        """Return the minimum value of a column."""
        return self.aggregate(agg_col, min)

    def aggregate_count(self, agg_col):
        """Return the count of values (non-null) in a column."""
        return self.aggregate(agg_col, len)

    def aggregate_mean(self, agg_col):
        """Return the mean of a column."""
        return self.aggregate(agg_col, lambda x: sum(x)/len(x))

    def aggregate_std(self, agg_col):
        """Return the standard deviation of a column."""
        import math
        def std_func(x):
            mean = sum(x)/len(x)
            return math.sqrt(sum((v - mean)**2 for v in x) / len(x))
        return self.aggregate(agg_col, std_func)
    
    def join(self, other, on, how="inner"):
        """
        Join two CSVDataFrames on a key column.

        Parameters
        ----------
        other : CSVDataFrame
            The right side of the join.
        on : str
            Column name to join on. Must exist in both DataFrames.
        how : str
            One of: "inner", "left", "right", "outer".

        Returns
        -------
        CSVDataFrame
            A new CSVDataFrame with joined data.
        """
        how = how.lower()
        if how not in ("inner", "left", "right", "outer"):
            raise ValueError(f"Unsupported join type: {how}")

        if on not in self.columns or on not in other.columns:
            raise KeyError(f"Join column '{on}' not found in both DataFrames.")

        # Build index for right table (hash join)
        index = {}
        right_keys = other.data[on]
        for j, key in enumerate(right_keys):
            if key not in index:
                index[key] = []
            index[key].append(j)

        # Output columns: all from left, plus non join columns from right
        new_columns = list(self.columns)
        for col in other.columns:
            if col != on:
                new_columns.append(col)

        # Prepare storage
        new_data = {col: [] for col in new_columns}

        # Track which right rows got matched (for right and outer joins)
        matched_right_indices = set()

        # Process left rows
        left_keys = self.data[on]
        for i, key in enumerate(left_keys):
            if key in index:
                # Matches exist on right
                for j in index[key]:
                    matched_right_indices.add(j)

                    # Left side columns
                    for col in self.columns:
                        new_data[col].append(self.data[col][i])

                    # Right side columns except join column
                    for col in other.columns:
                        if col != on:
                            new_data[col].append(other.data[col][j])
            else:
                # No match on right
                if how in ("left", "outer"):
                    # Keep left row, pad right with None
                    for col in self.columns:
                        new_data[col].append(self.data[col][i])
                    for col in other.columns:
                        if col != on:
                            new_data[col].append(None)
                # For pure inner or right join, we skip unmatched left rows

        # Handle unmatched right rows for right and outer joins
        if how in ("right", "outer"):
            num_right_rows = len(right_keys)
            for j in range(num_right_rows):
                if j not in matched_right_indices:
                    # No left row matched this right row
                    # Left side is None, right side is the row
                    for col in self.columns:
                        new_data[col].append(None)
                    for col in other.columns:
                        if col != on:
                            new_data[col].append(other.data[col][j])

        return CSVDataFrame.from_data(new_data)

    def inner_join(self, other, on):
        return self.join(other, on=on, how="inner")

    def left_join(self, other, on):
        return self.join(other, on=on, how="left")

    def right_join(self, other, on):
        return self.join(other, on=on, how="right")

    def outer_join(self, other, on):
        return self.join(other, on=on, how="outer")
    

