# import packages
import numpy as np
import pandas as pd

Prefix='../../../Data_CGPMSM/Kaggle/'

if __name__ == '__main__':

    # Check out the data
    data = pd.read_csv(Prefix + "input/road-weather-information-stations.csv", parse_dates=[2])
    data = data[['StationName', 'DateTime', 'RoadSurfaceTemperature', 'AirTemperature']]
    data.head(10)

    #print(data.columns)
    #data = data.head(1500)
    StationNames = data.StationName.unique()
    print(StationNames)
    for sn in StationNames:
        print(sn)
        df = data[data['StationName'].str.contains(sn)]
        #print(df.shape)
        df1 = df.sort_values(['DateTime'], ascending=True)
        name = Prefix + 'input/'+sn+'.csv'
        df1.to_csv(name, columns=['DateTime', 'RoadSurfaceTemperature', 'AirTemperature'], header=True, index=False)
        # name = Prefix + 'input/'+sn+'_RSTemp.csv'
        # df1.to_csv(name, columns=['DateTime', 'RoadSurfaceTemperature'], header=True, index=False)
        # name = Prefix + 'input/'+sn+'_ATemp.csv'
        # df1.to_csv(name, columns=['DateTime',  'AirTemperature'], header=True, index=False)