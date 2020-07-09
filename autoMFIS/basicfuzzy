from __future__ import division


def trimf(x, a, b, c):
    return max( min( (x-a)/(b-a), (c-x)/(c-b) ), 0 )

def trapmf(x, a, b, c, d):
  if (x <= a) or (x >= d):
    return 0
  elif (x >= b) and (x <= c):
    return 1
  elif (x > a) and (x < b):
    return (x-a)/(b-a)
  else:
    return (d-x)/(d-c)