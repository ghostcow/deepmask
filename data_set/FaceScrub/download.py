import urllib2
import re
import Image
import StringIO
import sys
import os
import time

# actors    - 6500
# actresses - 14000
inputFilePath = sys.argv[1] #'facescrub_unique.txt'
outputDir = 'images'
startLine = int(sys.argv[2]);
endLine = int(sys.argv[3]);
iLine = 0

with open(inputFilePath, "r") as f:
  for line in f:
    iLine += 1
    if (iLine < startLine):
      continue
    if (iLine > endLine):
      break
      
    x = line.split('	')
    if (x[0] == 'name'):
      continue
    
    name = x[0]
    id = int(x[2]) # face-id
    url = x[3]
    personDir = os.path.join(outputDir, name)
    filePath = os.path.join(personDir, '%06d.jpg'%(id))
    if not os.path.isfile(filePath):
      print('%d,%s,%d,%s'%(iLine,name,id,url))
      try:
	response = urllib2.urlopen(url, timeout=5)
	html = response.read()
      except KeyboardInterrupt:
        raise
      except:
	print "Unexpected error when downloading url:", sys.exc_info()[1] 
	continue
      
      if not os.path.isdir(personDir):
	os.makedirs(personDir)    
	
      try:
	stream = StringIO.StringIO(html)
	image = Image.open(stream)
	image.save(filePath)
      except KeyboardInterrupt:
        raise
      except:
	print "Unexpected error when saving image:", sys.exc_info()[1] 
	continue