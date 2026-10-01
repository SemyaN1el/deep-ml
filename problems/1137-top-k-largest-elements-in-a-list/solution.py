import heapq 

def top_three_largest(values):
    # values: list of numbers
    # return the three largest values in descending order
    heap = []

    for value in values:
        if len(heap) < 3:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap,value)

    return sorted(heap, reverse=True)