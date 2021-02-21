from pyimagesearch.cbir import ResultsMontage
from pyimagesearch.cbir import HSVDescriptor
from pyimagesearch.cbir import Searcher
import argparse
import imutils
import cv2
import json

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--index', required=True, help='path to where the features index will be stored')
ap.add_argument('-q', '--query', required=True, help='path to the query image')
ap.add_argument('-d', '--dataset', required=True, help='path to the original directory')
ap.add_argument('-r', '--relevant', required=True, help='path to relevant dictionary')
args = vars(ap.parse_args())

desc = HSVDescriptor((4, 6, 3))
montage = ResultsMontage((240, 320), 5, 20)
relevant = json.loads(open(args['relevant']).read())

queryFilename = args['query'][args['query'].rfind('/') + 1:]
queryRelevant = relevant[queryFilename]

# load the query image, display it, and describe it
print('[INFO] describing query...')
query = cv2.imread(args['query'])
cv2.imshow('Query', imutils.resize(query, width=320))
features = desc.describe(query)

# perform the search
print('[INFO] searching...')
searcher = Searcher(args['index'])
results = searcher.search(features, numResults=20)
# loop over the results
for (i, (score, resultID)) in enumerate(results):
	# load the result image and display it
	print('[INFO] {result_num}. {result} - {score:.2f}'.format(result_num=i + 1, result=resultID, score=score))
	result = cv2.imread('{}/{}'.format(args['dataset'], resultID))
	montage.addResult(result, text='#{}'.format(i + 1), highlight=resultID in queryRelevant)

# show the output image of results
cv2.imshow('Results', imutils.resize(montage.montage, height=700))
cv2.waitKey(0)