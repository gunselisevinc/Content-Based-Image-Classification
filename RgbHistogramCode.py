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
	modifiedHist=[]
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

#Validation with the whole image
def validationlevel1(kvalue,binvalue,image_listTrain,labels):
	dataset=[]
	hist_listTrain=[]
	for x in image_listTrain:
		histogram, bin_edges = np.histogram(x, bins=binvalue, range=(0, 256))
		hist2 = normalize(histogram)
		hist_listTrain.append(hist2)
	for h in hist_listTrain:
		dataset.append((h))

	knn = cv2.ml.KNearest_create()
	dataset_np = np.array(dataset, dtype=np.float32)
	labels_np = np.array(labels, dtype=np.float32)
	knn.train(dataset_np, None, labels_np)
	image_listValidation1=[]
	validationLabels=[]
	filelocation = "Validation" + "\\" + "*.jpg"  # open all png files
	for filename in glob.glob(filelocation):  # open all jpg files
		im3 = Image.open(filename).resize((500, 325))
		image_listValidation1.append(im3)
		if filename.find('cloudy') > -1:
			validationLabels.append(0)
		if filename.find('shine') > -1:
			validationLabels.append(1)
		if filename.find('sunrise') > -1:
			validationLabels.append(2)
	count=0
	correct=0
	for x in image_listValidation1:
		histogram1, bin_edges1 = np.histogram(x, bins=binvalue, range=(0, 256))
		test_hist = normalize(histogram1)
		input_data = [test_hist]
		input_data_np = np.array(input_data, dtype=np.float32)
		retval, results, neighbours, distances = knn.findNearest(input_data_np, k=kvalue)
		if int(retval) == int(validationLabels[count]):
			correct+=1
		count+=1
	accuracy=(correct/count )*100

	return accuracy

#Validation with 2x2 grid of image
def validationlevel2(kvalue,binvalue,image_listTrain,labels):
	dataset=[]
	hist_listTrain=[]
	for x in image_listTrain:
		hist2=gridSlicer(x,2,binvalue)
		hist_listTrain.append(hist2)
	for h in hist_listTrain:
		dataset.append((h))

	knn = cv2.ml.KNearest_create()
	dataset_np = np.array(dataset, dtype=np.float32)
	labels_np = np.array(labels, dtype=np.float32)
	knn.train(dataset_np, None, labels_np)

	image_listValidation1=[]
	validationLabels=[]
	filelocation = "Validation" + "\\" + "*.jpg"  # open all png files
	for filename in glob.glob(filelocation):  # open all jpg files
		im3 = Image.open(filename).resize((500, 325))
		image_listValidation1.append(im3)
		if filename.find('cloudy') > -1:
			validationLabels.append(0)
		if filename.find('shine') > -1:
			validationLabels.append(1)
		if filename.find('sunrise') > -1:
			validationLabels.append(2)
	count=0
	correct=0
	for x in image_listValidation1:
		test_hist = gridSlicer(x, 2, binvalue)
		input_data = [test_hist]
		input_data_np = np.array(input_data, dtype=np.float32)
		retval, results, neighbours, distances = knn.findNearest(input_data_np, k=kvalue)
		if int(retval) == int(validationLabels[count]):
			correct+=1
		count+=1
	accuracy=(correct/count )*100

	return accuracy

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

	image_listValidation1=[]
	validationLabels=[]
	filelocation = "Validation" + "\\" + "*.jpg"  # open all png files
	for filename in glob.glob(filelocation):  # open all jpg files
		im3 = Image.open(filename).resize((500, 325))
		image_listValidation1.append(im3)
		if filename.find('cloudy') > -1:
			validationLabels.append(0)
		if filename.find('shine') > -1:
			validationLabels.append(1)
		if filename.find('sunrise') > -1:
			validationLabels.append(2)
	count=0
	correct=0
	for x in image_listValidation1:
		test_hist = gridSlicer(x, 4, binvalue)
		input_data = [test_hist]
		input_data_np = np.array(input_data, dtype=np.float32)
		retval, results, neighbours, distances = knn.findNearest(input_data_np, k=kvalue)
		if int(retval) == int(validationLabels[count]):
			correct+=1
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


kvalues=[1,5,7,9]
binValues=[5,9,25,85,155,250] # we can investigate more
print("== LEVEL 1 ==")
for z in kvalues:
	for z1 in binValues:
		k=validationlevel1(z,z1,image_listTrain,labels)
		print("Accuracy : %"+str(k)+" with K value : "+str(z)+" and bin number : "+str(z1))
print("== LEVEL 2 ==")
for z in kvalues:
	for z1 in binValues:
		k = validationlevel2(z, z1, image_listTrain, labels)
		print("Accuracy : %" + str(k) + " with K value : " + str(z) + " and bin number : " + str(z1))
print("== LEVEL 3 ==")
for z in kvalues:
	for z1 in binValues:
		k = validationlevel3(z, z1, image_listTrain, labels)
		print("Accuracy : %" + str(k) + " with K value : " + str(z) + " and bin number : " + str(z1))

