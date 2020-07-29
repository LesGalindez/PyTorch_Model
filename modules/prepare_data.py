try:
    import os
    from pathlib import Path
    import pandas as pd
except Exception as e:
    print('PREPARATE DATASET MODEL MSSG: Not Import module: {}'.format(e))

class PreparationData():
    def __init__(self, config):
        self._config = config
        self._path_real_data = self._config['path_real_data']
        self._path_pickleFile = self._config['path_pickleFile']
        self._path_fileName = self._config['file_name']
    
    def data_processing(self):
        antispoofDF = pd.DataFrame()

        list_real = [file.path for file in os.scandir(self._path_real_data) if file.name.endswith('.jpg')]
        for imgPath in list_real:
            antispoofDF = antispoofDF.append({
                'image': str(imgPath),
                'real': 1
            }, ignore_index=True)

        dfName = self._path_pickleFile + self._path_fileName
        print(f'MSSG: Saving Dataframe to: {dfName}')
        antispoofDF.to_pickle(dfName)