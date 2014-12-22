import json
import os
import time
import requests
from PIL import Image
from StringIO import StringIO
from requests.exceptions import ConnectionError
import hashlib
import sys
import math

maxImages = 60 # Google will only return a max of 56 results.
numImagesPerPage = 4 # 4 images per page.

def go(query, path):
  """Download full size images from Google image search.
 
  Don't print or republish images without permission.
  I used this to train a learning algorithm.
  """
  BASE_URL = 'https://ajax.googleapis.com/ajax/services/search/images?'\
             'v=1.0&q=' + query + '&start=%d&imgtype=face'
 
  BASE_PATH = os.path.join(path, query)
 
  start = 0 # Google's start query string parameter for pagination.
  iImage = 0
  while start < maxImages: 
    try:
      r = requests.get(BASE_URL % start)
      if not os.path.exists(BASE_PATH):
	os.makedirs(BASE_PATH)	      
      print(r.text)
      for image_info in json.loads(r.text)['responseData']['results']:
	url = image_info['unescapedUrl']
	try:
	  image_r = requests.get(url)
	except ConnectionError, e:
	  print 'could not download %s' % url
	  continue
		  
	iImage = iImage + 1 
	# using md5 of the image content as name
	fileName = '%s_%04d' % (query,iImage) #hashlib.md5(image_r.content).hexdigest())
	try:
	  im = Image.open(StringIO(image_r.content))  
	  with open(os.path.join(BASE_PATH, '%s.jpg') % fileName, 'wb') as file:
	    im.save(file, format='JPEG', subsampling=0, quality=100)
	except IOError, e:
	  # Throw away some gifs...blegh.
	  print 'could not save %s' % url
	  continue
    except:
      print "Unexpected error:", sys.exc_info()[1] 
	
    print start
    start += numImagesPerPage 
 
    # Be nice to Google and they'll be nice back :)
    time.sleep(1)
 
# Example use
if __name__ == '__main__':
  outputPath = sys.argv[1]
  inputPath = sys.argv[2]
  maxImages = min(maxImages, int(sys.argv[3]))
  
  if (inputPath[-3:] == 'txt'):
    startLine = int(sys.argv[4]);
    endLine = int(sys.argv[5]);    
    
    # assume we got a txt with many persons names
    iLine = 0
    with open(inputPath, "r") as f:
      for line in f:
	iLine += 1
	
	if (iLine < startLine):
	  continue
	if (iLine > endLine):
	  break	
	
	if os.path.isdir(os.path.join(outputPath, line)):
	  continue
	print('%d - %s'%(iLine, line))
	go(line, outputPath)
  else:
    # assume we got person name
    print(inputPath)
    go(inputPath, outputPath)




