import numpy  as np


arrivals = [
    (1,8,2),
    (5,2,3),
    (2,3,9),
    (2,2,5),
    (1,7,1),
    (7,1,3),
    (3,4,6),
    (5,7,8),
    (4,1,7),
    (9,8,5),
    (1,5,2)
]


t = 0
tNextArrival = 2
arrIdx = 0
eosA = 5
eosB = np.nan
queueA = []
queueB = []
busyA = False
busyB = False

while(arrIdx < len(arrivals)):
    if t >= tNextArrival:
        tNextArrival += arrivals[arrIdx][0]

        if busyA and busyB:
            if len(queueA) <= len(queueB):
                queueA.append(arrivals[arrIdx][1])
            else:
                queueB.append(arrivals[arrIdx][2])
        elif busyA:
            queueB.append(arrivals[arrIdx][2])
        else:
            queueA.append(arrivals[arrIdx][1])
        arrIdx += 1

    if np.isnan(eosA) or t >= eosA:
        if queueA:
            eosA = t + queueA.pop()
        else:
            eosA = np.nan
    if np.isnan(eosB) or t >= eosB:
        if queueB:
            eosB = t + queueB.pop()
        else:
            eosB = np.nan

    busyA = not np.isnan(eosA)
    busyB = not np.isnan(eosB)

    t = np.nanmin([tNextArrival, eosA, eosB])

    print(tNextArrival, eosA, eosB, len(queueA), len(queueB))




