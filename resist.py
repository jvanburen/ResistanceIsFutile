values = { "black" : 0, "brown" : 1, "red" : 2, "orange" : 3, "yellow" : 4, "green" : 5, "blue" : 6, "violet" : 7, "grey" : 8, "white" : 9 }
tolerances = { "brown" : 1, "red" : 2, "green" : .5, "blue" : .25, "violet" : .1, "grey" :.05, "gold" : 5, "silver" : 10}
multiplier = { "black" : 1, "brown" : 10, "red" : 100, "orange" : 1000, "yellow" : 10000, "green" : 100000, "blue" : 10**6, "violet" : 10**7, "gold" : .1, "silver" : .01 }

# get_resistance : string list -> int * int
# REQUIRES: len(l) == 5 andalso the values in l are strings that are valid keys
# ENSURES: Proper 5 band parsed resistor coding
def get_resistance(l):
  if(len(l) != 5):
    raise ValueError
  return ((( values[l[0]] * 100 + values[l[1]] * 10 + values[l[2]]) * multiplier[l[3]]), tolerances[l[4]])


