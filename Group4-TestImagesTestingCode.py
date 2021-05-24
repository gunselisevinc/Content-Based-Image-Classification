import glob
from PIL import Image
import cv2
import numpy as np

##FUNCTION DECLARATION##

#Histogram Normalization
def normalize(hist):
	hist_np = np.array(hist, dtype=np.float32)
	hist_sum = sum(hist)
	hist_norm = hist_np / hist_sum
	return hist_norm.tolist()

#Dividing Histograms into n grids and attaching them end to end
def gridSlicer(im,n,binvalue):
	res_lt=[]
	imgwidth, imgheight = im.size
	height = imgheight // n
	width = imgwidth // n
	count=0
	for i in range(0, n):
		for j in range(0, n):
			box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
			a = im.crop(box)
			count+=1
			histogram, bin_edges = np.histogram(a, bins=binvalue, range=(0, 256))
			try:
				#res_lt = list( map (add, res_lt, normalize(histogram)))
				res_lt.extend(normalize(histogram))
			except:
				pass
	return res_lt

#Validation with 4x4 grid of image
def validationlevel3(kvalue,binvalue,image_listTrain,labels):
	dataset=[]
	hist_listTrain=[]
	for x in image_listTrain:
		hist2=gridSlicer(x,4,binvalue)
		#print(hist2)
		hist_listTrain.append(hist2)
	for h in hist_listTrain:
		dataset.append((h))
	knn = cv2.ml.KNearest_create()
	dataset_np = np.array(dataset, dtype=np.float32)
	labels_np = np.array(labels, dtype=np.float32)
	knn.train(dataset_np, None, labels_np)

	image_listTest=[]
	validationLabels=[]
	filelocation = "Test" + "\\" + "*.jpg"  # open all png files
	for filename in glob.glob(filelocation):  # open all jpg files
		im3 = Image.open(filename).resize((500, 325))
		image_listTest.append(im3)
		if filename.find('cloudy') > -1:
			validationLabels.append(0)
		if filename.find('shine') > -1:
			validationLabels.append(1)
		if filename.find('sunrise') > -1:
			validationLabels.append(2)
	count=0
	correct=0
	for x in image_listTest:
		test_hist = gridSlicer(x, 4, binvalue)
		input_data = [test_hist]
		input_data_np = np.array(input_data, dtype=np.float32)
		retval, results, neighbours, distances = knn.findNearest(input_data_np, k=kvalue)
		if int(retval) == int(validationLabels[count]):
			if retval < 1:
				print("It's a picture of a cloudy as correct (cloudy)")
			elif retval > 1:
				print("It's a picture of a sunrise  as correct  should be (sunrise)")
			else:
				print("It's a picture of a shine  as correct should be (shine)")
			correct+=1
		else:
			if int(validationLabels[count]) < 1:
				msg="cloudy"
			elif int(validationLabels[count]) > 1:
				msg="sunrise"
			else:
				msg="shine"


			if retval < 1:
				print("It's a picture of a cloudy as Incorrect! should be ("+msg+")")
			elif retval > 1:
				print("It's a picture of a sunrise  as Incorrect!  should be ("+msg+")")
			else:
				print("It's a picture of a shine  as Incorrect! should be ("+msg+")")

		count+=1
	accuracy=(correct/count )*100
	return accuracy

##MAIN##

dataset=[]
labels=[]
image_listTrain=[]
image_listTest=[]
image_listValidation=[]


filelocation="Train"+"\\"+"*.jpg" # open all png files

for filename in glob.glob(filelocation): # open all jpg files
	im = Image.open(filename).resize((500, 325))
	image_listTrain.append(im)
	if filename.find('cloudy') > -1:
		labels.append(0)
	if filename.find('shine') > -1:
		labels.append(1)
	if filename.find('sunrise') > -1:
		labels.append(2)

filelocation="Test"+"\\"+"*.jpg" # open all png files

for filename in glob.glob(filelocation): # open all jpg files
	im2 = Image.open(filename).resize((500, 325))
	image_listTest.append(im2)

gridSlicer(image_listTest[5],2,10)

hist_listValidation=[]
hist_listTest=[]
hist_listTrain=[]

#Testing
kvalues=[9]
binValues=[25] # we can investigate more

print("== BEST CASE WITH LEVEL3 K=9 BIN=25 RGB HISTOGRAM ==")
for z in kvalues:
	for z1 in binValues:
		k = validationlevel3(z, z1, image_listTrain, labels)
		print("Accuracy : %" + str(k) + " with K value : " + str(z) + " and bin number : " + str(z1))