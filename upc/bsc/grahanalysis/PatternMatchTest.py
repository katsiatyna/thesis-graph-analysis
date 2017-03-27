import re

p = re.compile('\(\d+, \d+\)')
matched = p.findall('((79, 127), (62, 63))')
print 'MATCHES ARE ' + str(matched)
for match in matched:
    p = re.compile('\d+')
    res = p.findall(match)
    print str(res)
