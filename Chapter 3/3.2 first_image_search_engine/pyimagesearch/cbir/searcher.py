from . import dists
import csv


class Searcher:
    def __init__(self, dbPath):
        # store the database path
        self.dbPath = dbPath

    def search(self, queryFeatures, numResults=10):
        # initialize the results dictonary
        results = {}

        # open the database for reading
        with open(self.dbPath) as f:
            reader = csv.reader(f)

            for row in reader:
                features = [float(x) for x in row[1:]]
                d = dists.chi2_distance(features, queryFeatures)

                results[row[0]] = d

            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])

        return results[:numResults]