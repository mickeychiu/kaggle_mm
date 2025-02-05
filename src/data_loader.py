import pandas as pd
import glob

verbosity = 0

class DataLoader:
    def __init__(self, root_path, paths):
        self.root = root_path
        self.paths = []
        for path in paths:
            self.paths.append(self.root + path)
        print('datapaths:\t',self.paths)
    
    def load_csvs(self):
        files = {}
        for path in self.paths:
            csvs = glob.glob(path + "*.csv")
            for csv in csvs:
                print("Reading " + csv + "...", end = "")
                files[csv.split('/')[-1]] = pd.read_csv(csv,encoding='cp1252')
                print(" done.")
        
        if verbosity:
          print(files)
        return files
