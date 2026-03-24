
"""
Joins together zipcode-level RUCA code, Population Demographic, Income, and Hospice Service area data with patient
hospice claims records based on their home zipcode

Compiles public sociodemographic data with the patient-level data
"""


import pandas as pd
import numpy as np





def getter(df_path: str) -> pd.DataFrame:
    return pd.read_csv(df_path)

def formatting(df, col1, col2=None):
    df[col1] = df[col1].astype(str).str.zfill(5)
    if col2:
        df[col2] = df[col2].astype(str).str.zfill(6)
        pass
    else:
        pass
    return df

def categorize_ruca(value):
    if 1.0 <= value <= 3.0:
        return "Urban"
    elif 4.0 <= value <= 6.0:
        return "Suburban"
    elif 7.0 <= value <= 10.0:
        return "Rural"
    else:
        return np.nan
def clean_ruca(df):
    df.rename(columns={"''ZIP_CODE''": 'ZIPCODE'}, inplace=True)
    df['ZIPCODE'] = df['ZIPCODE'].str.strip("''")
    df = df[['ZIPCODE', 'STATE', 'RUCA1', 'RUCA2']]
    df = formatting(df, 'ZIPCODE')
    df['ruca_category'] = df['RUCA1'].apply(categorize_ruca)
    return df

def clean_pop(df):
    df.rename(columns={"Geography": 'ZIPCODE'}, inplace=True)
    df['ZIPCODE'] = df['ZIPCODE'].str[9:]
    df = df[['ZIPCODE', 'Total']]
    df = formatting(df, 'ZIPCODE')
    return df

def clean_race(df):
    df.rename(columns={"GEO_ID": 'ZIPCODE',
                       'P9_001N': 'TotalPop',
                       'P9_002N': 'Hispanic',
                       'P9_005N': 'White',
                       'P9_006N': 'Black',
                       'P9_007N': 'American Indian',
                       'P9_008N': 'Asian'}, inplace=True)
    df['ZIPCODE'] = df['ZIPCODE'].str[9:]
    df = df[['ZIPCODE', 'TotalPop', 'Hispanic', 'White', 'Black', 'American Indian', 'Asian']]
    df = formatting(df, 'ZIPCODE')
    df = race_normalize(df)
    return df

def clean_income(df):
    df.rename(columns={'S1902_C03_001E': 'Income',
                       'NAME': 'ZIPCODE'}, inplace=True)
    df = df.iloc[1:]
    df['ZIPCODE'] = df['ZIPCODE'].str[6:]
    df = formatting(df, 'ZIPCODE')
    df['Income'] = df['Income'].replace(['N', "'-", "-"], np.nan)
    return df

def clean_hospice(df):
    df.rename(columns={'ZCTA5CE20': 'ZIPCODE'}, inplace=True)
    df = formatting(df, 'ZIPCODE')
    return df

def clean_service(df):
    df.rename(columns={'ZCTA5CE20': 'ZIPCODE'}, inplace=True)
    df.rename(columns={'FIRST_Region': 'Region'}, inplace=True)
    df = formatting(df, 'ZIPCODE')
    return df

def race_normalize(df):
    col_names = ['Black', 'White', 'Hispanic', 'Asian', 'American Indian']
    for col in col_names:
        df[col + '_pct'] = df[col] / df['TotalPop'] * 100
    return df

def main():
    ruca = getter('/drives/56219-Linux/56219dua/Project2018/1-Data/Census/RUCA2010zipcode.csv/RUCA2010zipcode.csv')
    ruca = clean_ruca(ruca)
    # print(ruca)


    # pop = getter('/drives/56219-Linux/56219dua/Project2018/1-Data/Census/2020Census/DECENNIALDHC2020.P1-Data.csv')
    # pop = clean_pop(pop)
    # print (pop)

    pop_race = getter('/drives/56219-Linux/56219dua/Project2018/1-Data/Census/2020Census/DECENNIALDHC2020.P9-Data.csv')
    pop_race = clean_race(pop_race)
    # print(pop_race)

    income = pd.read_csv('/drives/56219-Linux/56219dua/Project2018/1-Data/Census/2018CensusIncome/2018CensusIncome/ACSST5Y2018.S1902-Data.csv', usecols=['NAME', 'S1902_C03_001E'], low_memory=False)
    income = clean_income(income)
    # print(income)

    service = pd.read_csv('/drives/56219-Linux/56219dua/Project2018/1-Data/us_zcta_service_areas.csv', usecols=['ZCTA5CE20', 'MIN_ToBreak', 'FIRST_Region'])
    # service = pd.read_csv('/drives/56219-Linux/56219dua/Project2018/1-Data/Northeast_zcta_service.csv', usecols=['ZCTA5CE20', 'MIN_ToBreak',  'Region'])

    service = clean_service(service)
    # print(service)

    hospiceUtil = getter('/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/HospiceUtilCat.csv')
    hospiceUtil = clean_hospice(hospiceUtil)

    contextData = hospiceUtil.merge(ruca, on='ZIPCODE', how='left')
    contextData = contextData.merge(pop_race, on='ZIPCODE', how='left')
    contextData = contextData.merge(income, on='ZIPCODE', how='left')
    contextData = contextData.merge(service, on='ZIPCODE', how='left')

    # print(contextData)


    contextData.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/ContextHospiceUtil.csv',
        index=False)

    patientUtil = getter('/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/PatientUtilCat.csv')
    patientUtil = formatting(patientUtil, "ZIPCODE")

    patientContextData = patientUtil.merge(ruca, on='ZIPCODE', how='left')
    patientContextData = patientContextData.merge(pop_race, on='ZIPCODE', how='left')
    patientContextData = patientContextData.merge(income, on='ZIPCODE', how='left')
    patientContextData = patientContextData.merge(service, on='ZIPCODE', how='left')

    # patientContextData = patientContextData[patientContextData['RTI_RACE'].isin([1, 2, 4, 5])]


    patientContextData.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/ContextPatientUtil_All.csv',
        index=False)

if __name__ == '__main__':
    main()