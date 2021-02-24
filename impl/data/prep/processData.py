import argparse
import os
from data.prep.objectmesh import *

parser = argparse.ArgumentParser(description="Preprocessing data for CoMA")

parser.add_argument('--source', dest='source', type=str, required=True, help='path to the data directory')
parser.add_argument('--destination', dest='destination', type= str, required=True, default='source', help='path where the processed data will be saved')

def main():
    args = parser.parse_args()
    destination_path = args.destination
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    print("Preprocessing Slice Time Data")
    generateSlicedTimeDataSet(args.source, destination_path)

    print("preprocessing Expression Cross Validation")
    generateExpressionDataSet(args.source, destination_path)

if __name__  == '__main__':
    main()