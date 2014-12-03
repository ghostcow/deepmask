import urllib2
import re
import Image
import StringIO
import sys
import os
import time

inputFilePath = sys.argv[1]
outputDir = sys.argv[2]
startLine = int(sys.argv[3]);
endLine = int(sys.argv[4]);

# get more pictures except profile picture
getFewPics = True
numImages = 4

iLine = 0
with open(inputFilePath, "r") as f:
  for line in f:
    iLine += 1
    if (iLine < startLine):
      continue
    if (iLine > endLine):
      break
      
    x = line.split('	')
    name = x[1]
    url = x[2]
    filePath = os.path.join(outputDir, '%05d_%s.jpg'%(iLine, name))
    if not os.path.isfile(filePath):
      try:
	response = urllib2.urlopen(url)
	html = response.read()
      except:
	print "Unexpected error when loading main page:", sys.exc_info()[1] 
	time.sleep(5)
	continue
	
      # look for the profile image
      print('%d - %s'%(iLine, name))
      for match in re.findall(r'title=".* Picture"[\s]+src="[^"]+"',html, re.I):
	# print(match)
	x = re.split(r'src="([^"]+)"', match)
	imageUrl = x[1]
	try:
	  response2 = urllib2.urlopen(imageUrl)
	  html2 = response2.read()
	except:
	  print "Unexpected error when loading main image:", sys.exc_info()[1] 
	  time.sleep(5)
	  continue
	  
	stream = StringIO.StringIO(html2)
	image = Image.open(stream)
	image.save(filePath)
	break
      if not os.path.isfile(filePath):
	print('no profile image produced')
	#continue

    # success
    if getFewPics:
      try:
	response = urllib2.urlopen(url)
	html = response.read()
      except:
	print "Unexpected error when loading main page:", sys.exc_info()[1] 
	time.sleep(5)
	continue
	
      k1 = html.find(r'<div class="mediastrip">')
      if (k1 == -1):
	continue
      k2 = html.find(r'</div>', k1)
      if (k1 == -1):
	continue
      html = html[k1:k2]
      personDir = os.path.join(outputDir, '%05d_%s'%(iLine, name))
      if os.path.exists(personDir):
	continue
      else:
	os.makedirs(personDir)      
      print('%d - %s (more photos)'%(iLine, name))
      
      iImage = 1
      for pageMatch in re.findall(r'<a href="[^<>]+"', html, re.I):
	k = pageMatch.find('"')
	pageMatch = pageMatch[k+1:]
	k = pageMatch.find('"')
	imgPageUrl = pageMatch[:k]
	imgPageUrl = 'http://www.imdb.com' + imgPageUrl
	
	# open image page 
	try:
	  response2 = urllib2.urlopen(imgPageUrl)
	  html2 = response2.read()	
	except:
	  print "Unexpected error when loading other image page:", sys.exc_info()[1] 
	  time.sleep(5)
	  continue
	  
	k1 = html2.find(r'<table class="photo">')
	k2 = html2.find(r'src="', k1)
	k3 = html2.find(r'"', k2+5)
	imgUrl = html2[k2+5:k3]
	
	print(imgUrl)
	try:
	  response3 = urllib2.urlopen(imgUrl)
	  imgData = response3.read()
	except:
	  print "Unexpected error when loading other image:", sys.exc_info()[1] 
	  time.sleep(5)
	  continue
	  
	stream = StringIO.StringIO(imgData)
	image = Image.open(stream)
	fileName = '%05d_%s_%04d.jpg'%(iLine, name, iImage)
	image.save(os.path.join(personDir, fileName))
	
	iImage += + 1
	if (iImage > numImages):
	  break
