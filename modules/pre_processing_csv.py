try:
    import pandas as pd
    import shutil
    import os
except ImportError as e:
    print('PRE-PROCESSING DATA MSSG: Fail Import Module {}'.format(e))

class PreProcessingData():
    def __init__(self, config):
        self.config = config
        self.path = self.config['path']
        self.copy_pics = self.config['copy_pics']
        self.csvfiles_train = self.config['csvfiles_train']

    def csv_create(self, name_list, label_list, train=True):
        data = {
            'pic_name':name_list,
            'label': label_list
        }

        df = pd.DataFrame(data, columns = ['pic_name', 'label'])

        if train:
            df.to_csv('{}{}'.format(self.path, self.csvfiles_train))
            df = pd.read_csv('{}{}'.format(self.path, self.csvfiles_train))
    
    def data_list(self):
        label_real = []
        
        list_real = [file.name for file in os.scandir(self.path+'dataset/real/') if file.name.endswith('.jpg')]
        for i in enumerate(list_real):
            label_real.append('1')
            

        if list_real !=[] and label_real !=[]:
            self.csv_create(list_real, label_real, train=True)
        else:
            print('PRE-PROCESSING DATA MSSG: Empty list Train Dataset')
