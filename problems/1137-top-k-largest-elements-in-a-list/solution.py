import heapq 

def top_three_largest(values):
    # values: list of numbers
    # return the three largest values in descending order
    first = second = third = float('-inf')

    for value in values: 
        if value >= first: 
            first, second, third = value, first, second  
        elif value >= second:
            second, third = value, second 
        elif value > third:
            third = value 
    return [first, second, third][:len(values)]

    