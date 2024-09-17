from typing import List

def carPooling(trips: List[List[int]], capacity: int) -> bool:
    """
    There is a car with capacity empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

    You are given the integer capacity and an array trips where trips[i] = [numPassengersi, fromi, toi] indicates that the ith trip has numPassengersi passengers and the locations to pick them up and drop them off are fromi and toi respectively. The locations are given as the number of kilometers due east from the car's initial location.

    Return true if it is possible to pick up and drop off all passengers for all the given trips, or false otherwise.
    """
    # store counts of passengers at each location
    length_of_trip = [0 for _ in range(1001)]

    for passengers, start, end in trips:
        length_of_trip[start] += passengers # add passengers to start location
        length_of_trip[end] -= passengers # remove passengers from end location

    car_load = 0
    # count total passenger for each stop
    for i in range(1001):
        car_load += length_of_trip[i]
        if car_load > capacity:
            return False
    return True


trips = [[2,1,5],[3,3,7]];capacity = 4
assert carPooling(trips, capacity) == False
trips = [[2,1,5],[3,3,7]], capacity = 5
assert carPooling(trips, capacity) == True