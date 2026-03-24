"""
Pulls 2018 hospital records and compiles corresponding patient data

Summary example:

2018_patient_info.csv: list of unique patients and whether or not they used hospice


take eol18cacohortl6m.csv which has all patient info

check if in analytic_hos2018
- clean analytic by grouping by patientid,
- column to say y/n multiple providers
- column for y/n multiple discharge dates DSCHRGDT

"""

import numpy as np
import pandas as pd
from pathlib import Path


def get_patient_char_data(patient_char_data_path: str) -> pd.DataFrame:
    df_patients = pd.read_csv(patient_char_data_path)
    return df_patients


def get_hospital_data(hospital_path: str) -> pd.DataFrame:
    df_hospital = pd.read_csv(hospital_path)
    return df_hospital


def get_hospice_data(hospice_gen_path: str) -> pd.DataFrame:
    df_hospice_gen = pd.read_csv(hospice_gen_path)
    return df_hospice_gen


def filter_hospital_data(df_hospital: pd.DataFrame) -> pd.DataFrame:
    print("initial shape: ")
    print(df_hospital.shape)
    df_subset = df_hospital[df_hospital['claimType'] == "Hospice"]
    print("subset shape: ")
    print(df_subset.shape)
    return df_subset


def join_hospice_patient_data(df_patients: pd.DataFrame, df_hospice_util: pd.DataFrame) -> pd.DataFrame:
    merged_df = df_hospice_util.merge(df_patients[['BENE_ID', 'ZIPCODE', 'SEX', 'RTI_RACE']], left_on='patientID',
                                      right_on='BENE_ID', how='left')
    merged_df = merged_df.drop(columns=['BENE_ID'])
    print("merged_df shape: ")
    print(merged_df.shape)

    return merged_df


def remove_duplicates(df_merged: pd.DataFrame) -> pd.DataFrame:
    new_merged = df_merged.drop_duplicates(subset=['patientID', 'MDID', 'PLCSRVC'])
    return new_merged


def join_hospice_gen_data(df_util: pd.DataFrame, df_hospice_gen: pd.DataFrame) -> pd.DataFrame:
    df_util['PROVIDER'] = df_util['PROVIDER'].astype(str).str.zfill(6)
    df_hospice_gen['CMS Certification Number (CCN)'] = df_hospice_gen['CMS Certification Number (CCN)'].astype(str).str.zfill(6)
    merged_df = df_util.merge(df_hospice_gen, left_on='PROVIDER',
                              right_on='CMS Certification Number (CCN)', how='left')
    print("merged_df shape: ")
    print(merged_df.shape)

    return merged_df


def classify_service(x):
    values = set(x.dropna().unique())
    if not values:
        return 0
    elif values == {1}:
        return 1
    elif values == {2}:
        return 2
    elif values == {1,2}:
        return 3
    else:
        return -1

def categorize_hospice_patients(df):
    summary = df.groupby('BENE_ID').agg(
        n_providers=('PROVIDER', 'nunique'),
        n_discharges=('DSCHRGDT', 'nunique')
    ).reset_index()

    summary['multiple_providers'] = summary['n_providers'] > 1
    summary['multiple_discharges'] = summary['n_discharges'] > 1

    df = df.merge(summary, on='BENE_ID', how='left')
    df = df[['BENE_ID', 'PROVIDER', 'TYPESRVC', 'multiple_providers', 'multiple_discharges']]
    df['service_loc'] = df.groupby('BENE_ID')['TYPESRVC'].transform(classify_service)
    return df

def find_hospice_patient_matches(df_all, df):
    df_merged = df_all.merge(df, on='BENE_ID', how='left')

    df_merged['hospice_use'] = df_merged['multiple_providers'].notna()

    df_merged = df_merged.drop_duplicates(subset=['BENE_ID', 'PROVIDER'], keep='first')

    df_merged = df_merged[['BENE_ID', 'PROVIDER', 'SEX', 'RACE', 'ZIPCODE', 'agedgrp5', 'RTI_RACE', 'hsa', 'hsalabel',
                           'hrr', 'hrrlabel','STATE', 'mismatch_state',
                           'multiple_providers', 'multiple_discharges', 'hospice_use', 'TYPESRVC', 'service_loc']]

    df_merged['hospice_category'] = df_merged.apply(categorize, axis=1)

    return df_merged

def categorize(row):
    if not row['hospice_use']:
        return 0
    elif not row['multiple_providers'] and not row['multiple_discharges']:
        return 1
    elif not row['multiple_providers'] and row['multiple_discharges']:
        return 2
    elif row['multiple_providers'] and row['multiple_discharges']:
        return 3
    elif row['multiple_providers'] and not row['multiple_discharges']:
        return 4
    else:
        return -1

def main():
    # all patient info
    save_path1 = Path('/drives/56219-Linux/56219dua/Project2018/1-Data/2018/eol18cacohortl6m.csv')
    df_patients = get_patient_char_data(save_path1)

    # general hospice info
    save_path3 = Path('/drives/56219-Linux/56219dua/Project2017/1-Data/Hospice_General-Information_JUL2021.csv')
    df_hospice_gen = get_hospice_data(save_path3)
    print(df_hospice_gen['CMS Certification Number (CCN)'].isna().count())
    df_hospice_cleaned = df_hospice_gen[
        ~df_hospice_gen['CMS Certification Number (CCN)'].str.contains(r'[a-zA-Z]', na=False)]

    df_hospice_cleaned['CMS Certification Number (CCN)'] = df_hospice_cleaned['CMS Certification Number (CCN)'].astype(
        int)

    # hospice patient data
    save_path4 = Path('/drives/56219-Linux/56219dua/Project2018/1-Data/2018/analytic_hos2018.csv')
    df_hospice_bene = get_hospice_data(save_path4)

    df_hospice_bene = categorize_hospice_patients(df_hospice_bene)

    # join hospice patient data with general patient data
    all_data = find_hospice_patient_matches(df_patients, df_hospice_bene)

    final_data = join_hospice_gen_data(all_data, df_hospice_cleaned)
    # all_data['PROVIDER'] = all_data['PROVIDER'].astype("object")
    # df_hospice_cleaned['CMS Certification Number (CCN)'] = df_hospice_cleaned['CMS Certification Number (CCN)'].astype("object")
    # merged_df = all_data.merge(df_hospice_cleaned, left_on='PROVIDER',
    #                           right_on='CMS Certification Number (CCN)', how='left')


    #
    # print(final_data['hospice_category'].value_counts())
    # print(final_data['TYPESRVC'].value_counts())


    final_data.to_csv(
        f'/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/1-PatientData.csv',
        index=False)


if __name__ == '__main__':
    main()

