import os
from data.prep.objectmesh import *

SOURCE = "/media/oole/Storage/Msc/example-data/registered-data/COMA_data"
DESTINATION = "/media/oole/Storage/Msc/processed-data"

def main():

    if not os.path.exists(DESTINATION):
        os.makedirs(DESTINATION)

    print("Preprocessing Slice Time Data")
    generateSlicedTimeDataSet(SOURCE, DESTINATION)

    print("preprocessing Expression Cross Validation")
    generateExpressionDataSet(SOURCE, DESTINATION)

if __name__  == '__main__':
    main()