while(True):
    hour = input('Hour?\n').zfill(2)
    min = input('Minute?\n').zfill(2)
    sec = str(int(input('Seconds?\n')) + 1).zfill(2)

    if int(sec) >= 60:
        sec = '00'
        min = str(int(min) + 1).zfill(2)

    if int(min) >= 60:
        min = '00'
        hour = str(int(hour) + 1).zfill(2)

    if int(hour) >= 24:
        hour = '00'

    print(hour + ':' + min + ':' + sec + '\n'*2)