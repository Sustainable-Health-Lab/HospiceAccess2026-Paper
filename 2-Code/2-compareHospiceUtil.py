
"""
Pull 2018_Zipdata
Pull utilization data
- group by hospice count for each race category
- creates df with each zipcode, checks if there's a utilization record
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd





def getter(df_path: str) -> pd.DataFrame:
    return pd.read_csv(df_path)

def getter_shp(df_path: str):
    return gpd.read_file(df_path)


def group_nonpatients(df):
    df_nonhos = df[df['hospice_use'] == False]
    # df_nonhos = df_nonhos.groupby(['ZIPCODE', 'is_covered']).size().reset_index(name='n_nonhosp')
    df_nonhos = df_nonhos.groupby(['ZIPCODE']).size().reset_index(name='n_nonhosp')

    return df_nonhos


def group_nonpatients_race(df):
    df_nonhos = df[df['hospice_use'] == False]
    df_nonhos = df_nonhos.groupby(['ZIPCODE', 'RTI_RACE']).size().reset_index(name='n_nonhosp')
    pivot_hos = df_nonhos.pivot(
        index=['ZIPCODE'],
        columns='RTI_RACE',
        values='n_nonhosp'
    ).fillna(0).astype(int)
    # pivot_hos = pivot_hos.reset_index()
    return pivot_hos


def group_patients(df):
    df_hos = df[df['hospice_use'] == True]
    # df_hos = df_hos.groupby(['ZIPCODE', 'PROVIDER', 'is_covered']).size().reset_index(name='n_hos')
    df_hos = df_hos.groupby(['ZIPCODE', 'PROVIDER']).size().reset_index(name='n_hos')

    return df_hos


def group_patients_tot(df):
    df_hos = df[df['hospice_use'] == True]
    df_hos = df_hos.groupby(['ZIPCODE']).size().reset_index(name='n_hos')
    return df_hos


def group_patients_race(df):
    df_hos = df[df['hospice_use'] == True]
    df_hos = df_hos.groupby(['ZIPCODE', 'PROVIDER', 'RTI_RACE']).size().reset_index(name='n_hos')

    pivot_hos = df_hos.pivot(
        index=['ZIPCODE', 'PROVIDER'],
        columns='RTI_RACE',
        values='n_hos'
    ).fillna(0).astype(int)
    pivot_hos = pivot_hos.reset_index()
    return pivot_hos


def group_patients_race_tot(df):
    df_hos = df[df['hospice_use'] == True]
    df_hos = df_hos.groupby(['ZIPCODE', 'RTI_RACE']).size().reset_index(name='n_hos')

    pivot_hos = df_hos.pivot(
        index=['ZIPCODE'],
        columns='RTI_RACE',
        values='n_hos'
    ).fillna(0).astype(int)
    pivot_hos = pivot_hos.reset_index()
    return pivot_hos


def formatting(df, col1, col2=None):
    df[col1] = df[col1].astype(str).str.zfill(5)
    if col2:
        df[col2] = df[col2].astype(str).str.zfill(6)
        pass
    else:
        pass
    return df


def zctas_covered(df):
    df_zctas = (
        df.groupby('Zip Code')['CMS Certification Number (CCN)']
        .apply(list)
        .reset_index()
        .rename(columns={'CMS Certification Number (CCN)': 'hospices'})
    )
    return df_zctas


def combine_all(df_shp, zctas, served, not_served, served_race, not_served_race):
    df_shp = df_shp.merge(served, left_on="ZCTA5CE20", right_on="ZIPCODE", how="left")
    # served1 = served.drop(columns=['is_covered'], errors='ignore')

    # not_served1 = not_served.drop(columns=['is_covered'], errors='ignore')
    # served_race1 = served_race.drop(columns=['is_covered'], errors='ignore')
    # not_served_race1 = not_served_race.drop(columns=['is_covered'], errors='ignore')
    df_shp = df_shp.merge(not_served, left_on="ZCTA5CE20", right_on="ZIPCODE", how="left")
    df_shp = formatting(df_shp, 'ZCTA5CE20')
    df_shp['is_covered'] = df_shp['ZCTA5CE20'].isin(zctas['Zip Code'])



    df_shp = df_shp.merge(served_race, left_on="ZCTA5CE20", right_on="ZIPCODE", how="left")


    df_shp = df_shp.merge(not_served_race, left_on="ZCTA5CE20", right_on="ZIPCODE", how="left", suffixes=('_served', '_notserved'))

    df_shp = df_shp.fillna(0).astype(int)
    df_shp = df_shp.drop(columns=[col for col in df_shp.columns if 'ZIPCODE' in col])

    return df_shp

def join_with_shp(zip_shp, df_all, zctas_serviced):
    zip_shp = zip_shp.merge(df_all, left_on="ZCTA5CE20", right_on="ZIPCODE", how="left")
    zip_shp['is_covered'] = zip_shp['ZCTA5CE20'].isin(zctas_serviced['Zip Code'])
    return zip_shp


def zipcode_categorize(row):
    if row['n_hos'] == 0 and row['n_nonhosp'] == 0:
        return 0
    elif row['is_covered'] and row['n_hos'] > 0:
        return 1 # served and used
    elif not row['is_covered'] and row['n_hos'] > 0:
        return 2 # not served and used
    elif row['is_covered'] and row['n_hos'] == 0:
        return 3 # served and not used
    else:
        return 4 # not served and not used

def add_totals(df):
    df['n_total'] = df['n_hos'] + df['n_nonhosp']
    return df

def patient_categorize(row):
    if row['is_covered'] and row['hospice_use']:
        return 1 # served and used
    elif not row['is_covered'] and row['hospice_use']:
        return 2 # not served and used
    elif row['is_covered'] and not row['hospice_use']:
        return 3 # served and not used
    else:
        return 4 # not served and not used




def main():
    # get PatientData
    save_path1 = Path('/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-PatientData.csv')
    df_patients = getter(save_path1)
    df_patients = formatting(df_patients, 'ZIPCODE', 'PROVIDER' )

    # get data on hospice coverage zones
    # group data by zcta
    save_path3 = Path('/drives/56219-Linux/56219dua/Project2018/1-Data/2018Zipdata.csv/2018Zipdata.csv')
    df_hospice_zips = getter(save_path3)
    df_hospice_zips = formatting(df_hospice_zips,  'Zip Code', 'CMS Certification Number (CCN)')
    zctas_serviced = zctas_covered(df_hospice_zips)

    # is_covered columns
    df_patients['is_covered'] = df_patients['ZIPCODE'].isin(zctas_serviced['Zip Code'])

    # group by patient category
    df_nonhos = group_nonpatients(df_patients)
    df_hos = group_patients(df_patients)
    df_hos_tot = group_patients_tot(df_patients)

    df_nonhos_race = group_nonpatients_race(df_patients)
    df_hos_race = group_patients_race(df_patients)
    df_hos_race_tot = group_patients_race_tot(df_patients)

    print(df_hos_tot.dtypes)
    print(df_nonhos.dtypes)




    df_shp = getter_shp('../1-Data/shpfiles/tl_2020_us_zcta520/tl_2020_us_zcta520/tl_2020_us_zcta520.shp')
    df_shp = df_shp[['ZCTA5CE20']].copy()
    df_shp = df_shp.drop_duplicates()
    df_shp = formatting(df_shp, 'ZCTA5CE20')

    # df_all = df_all[['ZIPCODE', 'n_hos', 'n_nonhosp']].copy()
    df_all = combine_all(df_shp, zctas_serviced, df_hos_tot, df_nonhos, df_hos_race_tot, df_nonhos_race)
    df_all = df_all.fillna(0)

    df_all.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/HospiceUtil.csv',
        index=False)

    df_all['coverage_use'] = df_all.apply(zipcode_categorize, axis=1)
    df_all = add_totals(df_all)

    df_all.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/HospiceUtilCat.csv',
            index=False)


    df_patients['coverage_use'] = df_patients.apply(patient_categorize, axis=1)
    df_patients.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/PatientUtilCat.csv',
            index=False)







    df_nonhos.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-NotServed.csv',
        index=False)

    df_hos.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-Served.csv',
        index=False)

    df_hos_tot.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-ServedSum.csv',
        index=False)

    df_nonhos_race.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-NotServedRace.csv',
        index=False)

    df_hos_race.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-ServedRace.csv',
        index=False)

    df_hos_race_tot.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-ServedSumRace.csv',
        index=False)

    zctas_serviced.to_csv(
        '/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/zctaServiced.csv',
        index=False)




if __name__ == '__main__':
    main()

