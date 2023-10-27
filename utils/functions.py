def getInsertionIndex(element, elements, key=lambda x: x):
    low = 0
    high = len(elements) - 1
    while low <= high:
        mid = (low + high)//2
        if key(elements[mid]) == key(element):
            # if there are duplicates in the list
            if mid + 1 < len(elements) and key(elements[mid+1]) == key(element):
                low = mid + 1
            else:
                return mid + 1
        elif key(elements[mid]) < key(element):
            low = mid + 1
        else:
            high = mid - 1
    return low