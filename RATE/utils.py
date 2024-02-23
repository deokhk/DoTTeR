"""
Util function for summary generation
"""


def getrank_column_direction(extracted_values, vtypes, ctype):
    # Get the rank of the values in the column with the same value type
    # Return List of (rank, index) pair and total number of values with same type in the column
    # The larger the number, the lower the rank
    # The rank starts from 1!

    index = [i for i in range(len(extracted_values))]
    filtered_values = []
    for (idx, value, vtype) in zip(index, extracted_values, vtypes):
        if vtype == ctype or (vtype == "range" and ctype == "numeric"):
            filtered_values.append((idx, value))

    if ctype == "numeric":
        filtered_values.sort(key=lambda x: x[1], reverse=True)
    elif ctype == "date":
        try:
            filtered_values.sort(key=lambda x: x[1], reverse=True)
        except (TypeError, ValueError):
            return -1 
    else:
        raise ValueError("Invalid column type")
    cv_ranks = []
    for rank, (idx, value) in enumerate(filtered_values):
        cv_ranks.append((idx, rank+1)) # Here, the index is the original index in the column

    return cv_ranks

