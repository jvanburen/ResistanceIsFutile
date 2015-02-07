import math

values = { "black" : 0, "brown" : 1, "red" : 2, "orange" : 3, "yellow" : 4,
          "green" : 5, "blue" : 6, "violet" : 7, "grey" : 8, "white" : 9 }
tolerances = { "brown" : 1, "red" : 2, "green" : .5, "blue" : .25, "violet" : .1,
              "grey" :.05, "gold" : 5, "silver" : 10}
multiplier = { "black" : 1, "brown" : 10, "red" : 100, "orange" : 1000, "yellow" : 10000,
              "green" : 100000, "blue" : 10**6, "violet" : 10**7, "gold" : .1, "silver" : .01 }

# get_resistance : string list -> int * int
# REQUIRES: len(l) == 5 andalso the values in l are strings that are valid keys
# ENSURES: Proper 5 band parsed resistor coding
def get_resistance(l):
  if(len(l) != 5):
    raise ValueError
  return (((values[l[0]] * 100 + values[l[1]] * 10 + values[l[2]]) * multiplier[l[3]]), tolerances[l[4]])

points = { "brown" : (22.0, 4.0, 180.0),
           "red"   : (0.0, 3.5, 255.0),
           "orange": (11.0, 2.0, 255.0),
           "yellow" : (26.0, 2.0, 255.0),
           "green" : (56.0, 10.0, 255.0),
           "blue" : (119.0, 7.0, 255.0),
           "violet" : (170.0, 3.0, 255.0)
         }

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def score(a, b):
  h, s, v = a
  mu, sigma, lightness = b
  score = 1 - abs(lightness - (s + v)/2) / 255.0
  return score * max(normpdf(h, mu, sigma), normpdf((h - 180), mu, sigma))

def match(color):
  h, s, v = color
  if s < 60:
    if v > 200:
      return [ ("white", -1) ]
    elif v < 65:
      return [ ("black", -1) ]
    else:
      return [ ("grey", -1) ]
  else:
    d = []
    for k,v in points.items():
      d.append((k, score(color, v)))
    d.sort(key=lambda x: x[1], reverse=True)
    return d
