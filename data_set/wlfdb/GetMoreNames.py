import urllib2
import re
import sys

year = int(sys.argv[1]);
outputFilePath = '../names/people_%d.txt'%year
print(outputFilePath)

getListUrl = 'http://www.imdb.com/search/name?birth_date=%d,%d&count=100'
getListMoreUrl = 'http://www.imdb.com/search/name?birth_date=%d,%d&count=100&start=%d';
numPeoplePerPage = 100;

k = 1
with open(outputFilePath, 'w') as the_file:  
  url = getListUrl%(year, year + 1);
  while True:
    response = urllib2.urlopen(url)
    html = response.read()

    # regexStr = '<a href="/name/nm0000124/">Jennifer Connelly</a>'
    numPersons = 0
    for match in re.findall(r'<td class="name">\s*<a href="/name/nm[0-9]+/">[^<>/="]+</a>',html, re.I):
      x = re.split(r'<a href="(.*)">(.*)</a>', match)
      numPersons += 1
      #print('%s,%s'%(x[2], x[1]))
      the_file.write('%s,%s\n'%(x[2], x[1]))
    print('%d -> %d'%(k,numPersons))
    if (numPersons == 0):
      break
    k = k + numPeoplePerPage
    url = getListMoreUrl%(year, year + 1, k)