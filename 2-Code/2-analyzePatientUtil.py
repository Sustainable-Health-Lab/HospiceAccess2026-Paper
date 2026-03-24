"""
Prepares patient-level data for analysis
Outputs Analysis_ContextPatientUtil_test.csv

Includes code to run logistic regressions in python and R -> NOT used in final analysis
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import seaborn as sns
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri




# custom_lib_path = "/drives/drive1/home/f0055vd/R/x86_64-redhat-linux-gnu-library/4.3"
#
# robjects.r(f'.libPaths(c("{custom_lib_path}", .libPaths()))')
#
# print(robjects.r('.libPaths()'))
#
# robjects.r('library(lme4)')
#
#
# pandas2ri.activate()






def getter(df_path: str) -> pd.DataFrame:
    return pd.read_csv(df_path)


def formatting(df, col2, col1=None):
    df[col2] = df[col2].astype(str).str.zfill(5)
    if col1:
        df[col1] = df[col1].astype(str).str.zfill(6)
        pass
    else:
        pass
    return df

def fit_logistic_regression(state_df, state):

    model = smf.logit("hospice_use ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income",

        data=state_df).fit()

    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_patient_logistic.txt', 'w') as f:
        f.write(model.summary().as_text())

    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']
    errors = (conf['upper'] - conf['lower']) /2

    coefs.plot(kind='bar', yerr=errors, figsize=(10, 6), legend=False)
    plt.title(f'{state} Logistic Regression Coefficients with 95% CI')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_logistic_coefs.png", dpi=300)
    plt.close()

    return model.summary()

def fit_multinom_logistic_regression(outcome, state_df, state):
    print(f'{outcome} for {state}')

    model = smf.mnlogit(f"{outcome} ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income",
        data=state_df).fit()

    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_{outcome}_multinom.txt', 'w') as f:
        f.write(model.summary().as_text())

    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']
    errors = (conf['upper'] - conf['lower']) / 2

    coefs.plot(kind='bar', yerr=errors, figsize=(10, 6), legend=False)
    plt.title(f'{state} {outcome} Multinom Regression Coefficients with 95% CI')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_{outcome}_coefs.png", dpi=300)
    plt.close()

    return model.summary()

# def fit_multinom_logistic_regression(outcome, state_df, state):
#
#     model = smf.mnlogit(f"{outcome} ~ agedgrp5 + SEX + is_covered + ruca_category + is_white + is_black + is_asian + is_hispanic + is_other + Black_pct + White_pct + Hispanic_pct + Asian_pct + Native_pct + lt30k_60k + gt60k_100k + gt100k",
#         data=state_df).fit()
#
#     with open(f'../4-Analysis/3-Analysis/Gabrielle/summary_{state}_{outcome}_multinom.txt', 'w') as f:
#         f.write(model.summary().as_text())
#
#     return model.summary()


def fit_logistic_regression_race(race_df, race):
    model = smf.logit("hospice_use ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income",

        data=race_df).fit()

    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{race}_patient_logistic.txt', 'w') as f:
        f.write(model.summary().as_text())

    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']
    errors = (conf['upper'] - conf['lower']) /2

    coefs.plot(kind='bar', yerr=errors, figsize=(10, 6), legend=False)
    plt.title(f'{race} Logistic Regression Coefficients with 95% CI')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{race}_logistic_coefs.png", dpi=300)
    plt.close()


    return model.summary()
def fit_logistic_regression_service(state_df, state):

    model = smf.logit("hospice_use ~ is_female + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct +is15min + is30min +  is60min + Income",

        data=state_df).fit()

    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_patient_logistic.txt', 'w') as f:
        f.write(model.summary().as_text())

    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']
    errors = (conf['upper'] - conf['lower']) /2

    coefs.plot(kind='bar', yerr=errors, figsize=(10, 6), legend=False)
    plt.title(f'{state} Logistic Regression Coefficients with 95% CI')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_logistic_coefs.png", dpi=300)
    plt.close()

    return model.summary()

def fit_logistic_regression_service1(state_df, title, state, region_color):

    rename_dict = {
        'const': "Intercept",
        'is_female': 'Sex (Female)',
        # 'agedgrp5_2': '70-74 Age Group',
        # 'agedgrp5_3': '75-79 Age Group',
        # 'agedgrp5_4': '80-84 Age Group',
        # 'agedgrp5_5': '85+ Age Group',
        'agedgrp5': 'Age Group',
        'is_black': 'Black',
        'is_asian': 'Asian',
        'is_hispanic': 'Hispanic',
        'sex_black': 'Sex x Black',
        'sex_asian': 'Sex x Asian',
        'sex_hispanic': 'Sex x Hispanic',
        'urban_black': 'Urban x Black',
        'urban_asian': 'Urban x Asian',
        'urban_hispanic': 'Urban x Hispanic',
        'suburban_black': 'Suburban x Black',
        'suburban_asian': 'Suburban x Asian',
        'suburban_hispanic': 'Suburban x Hispanic',
        'is15min_black': 'Within 15 min x Black',
        'is15min_asian': 'Within 15 min x Asian',
        'is15min_hispanic': 'Within 15 min x Hispanic',
        'is30min_black': 'Within 15-30 min x Black',
        'is30min_asian': 'Within 15-30 min x Asian',
        'is30min_hispanic': 'Within 15-30 min x Hispanic',
        'is60min_black': 'Within 30-60 min x Black',
        'is60min_asian': 'Within 30-60 min x Asian',
        'is60min_hispanic': 'Within 30-60 min x Hispanic',
        'Black_pct_black': '% Black in ZIP x Black',
        'Black_pct_asian': '% Black in ZIP x Asian',
        'Black_pct_hispanic': '% Black in ZIP x Hispanic',
        'Hispanic_pct_black': '% Hispanic in ZIP x Black',
        'Hispanic_pct_asian': '% Hispanic in ZIP x Asian',
        'Hispanic_pct_hispanic': '% Hispanic in ZIP x Hispanic',
        'Asian_pct_black': '% Asian in ZIP x Black',
        'Asian_pct_asian': '% Asian in ZIP x Asian',
        'Asian_pct_hispanic': '% Asian in ZIP x Hispanic',
        'White_pct_black': '% White in ZIP x Black',
        'White_pct_asian': '% White in ZIP x Asian',
        'White_pct_hispanic': '% White in ZIP x Hispanic',

        'age_black': 'Age x Black',
        'age_asian': 'Age x Asian',
        'age_hispanic': 'Age x Hispanic',
        'income_black': 'Income x Black',
        'income_asian': 'Income x Asian',
        'income_hispanic': 'Income x Hispanic',
        'hsa': 'Hospital Service Areas',
        'Black_pct': '% Black in ZIP',
        'White_pct': '% White in ZIP',
        'Hispanic_pct': '% Hispanic in ZIP',
        'Asian_pct': '% Asian in ZIP',
        'is15min': 'Within 15 Min of Hospice',
        'is30min': 'Within 15-30 Min of Hospice',
        'is60min': 'Within 30-60 Min of Hospice',
        # 'morethan60': 'More than 60 Min from Hospice',
        'Income': 'Mean Income by ZIP'
    }

    # x_cols = ['SEX', 'Suburban', 'Urban', 'agedgrp5_2', 'agedgrp5_3', 'agedgrp5_4', 'agedgrp5_5', 'is_black', 'is_asian', 'is_hispanic',
    #           'Black_pct', 'White_pct', 'Hispanic_pct', 'Asian_pct', 'is30min',  'is60min', 'morethan60', 'Income']
    x_cols = ['Income', 'Hispanic_pct', 'Asian_pct', 'Black_pct', 'White_pct', 'is15min', 'is30min', 'is60min', 'Suburban',
              'Urban',
              'is_hispanic', 'is_asian', 'is_black',
              'sex_black','sex_asian', 'sex_hispanic',
              'urban_black', 'urban_asian', 'urban_hispanic',
              'suburban_black', 'suburban_asian', 'suburban_hispanic',
              'is15min_black', 'is15min_asian', 'is15min_hispanic',

              'is30min_black', 'is30min_asian', 'is30min_hispanic',
              'is60min_black', 'is60min_asian', 'is60min_hispanic',

              'age_black', 'age_asian', 'age_hispanic',
              'income_black', 'income_asian', 'income_hispanic',
              'Hispanic_pct_black', 'Hispanic_pct_asian', 'Hispanic_pct_hispanic',
              'Asian_pct_black', 'Asian_pct_asian', 'Asian_pct_hispanic',
              'Black_pct_black', 'Black_pct_asian', 'Black_pct_hispanic',
              'White_pct_black', 'White_pct_asian', 'White_pct_hispanic',

              # 'agedgrp5_2', 'agedgrp5_3', 'agedgrp5_4', 'agedgrp5_5',
              'agedgrp5',
              'is_female']
    y_col = 'hospice_use'

    X = state_df[x_cols]
    y = state_df[y_col]

    X = sm.add_constant(X)

    X = X.astype(float)
    print(X.isnull().sum())

    logit_model = sm.Logit(y, X)
    logit_results = logit_model.fit(cov_type='cluster',
                                    cov_kwds={'groups': state_df['hsa']})

    print(logit_results.summary())

    marginal_effects = logit_results.get_margeff(method='dydx', at='overall')
    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_patient_logistic.txt', 'w') as f:
        f.write(logit_results.summary().as_text())
        f.write(marginal_effects.summary().as_text())


    R = logit_results.summary().tables[1].data
    R = pd.DataFrame(R)
    R.columns = R.iloc[0]

    R = R.iloc[1:]
    R[R.columns[1:]] = R[R.columns[1:]].astype(float)
    R.columns = ['vari', 'coef', 'std_err', 't', 'p_val', 'low_b', 'up_b']
    R['vari_cleaned'] = R['vari'].replace(rename_dict)
    # R = R.sort_values('coef', ascending=True)
    R = R.reset_index(drop=True)
    # print(R['vari_cleaned'].values)

    custom_order = ["Intercept", 'Sex (Female)',
                    # '70-74 Age Group', '75-79 Age Group', '80-84 Age Group', '85+ Age Group',
                    'Age Group',
                    'Black', 'Asian', 'Hispanic',
                    'Sex x Black',
                    'Sex x Asian', 'Sex x Hispanic',
                    'Age x Black',
                    'Age x Asian', 'Age x Hispanic',
                    'Income x Black',
                    'Income x Asian', 'Income x Hispanic',
                    'Urban x Black', 'Urban x Asian','Urban x Hispanic', 'Suburban x Black',
                    'Suburban x Asian', 'Suburban x Hispanic',
                    'Within 15 min x Black',
                    'Within 15 min x Asian', 'Within 15 min x Hispanic',
                    'Within 15-30 min x Black',
                    'Within 15-30 min x Asian', 'Within 15-30 min x Hispanic',
                    'Within 30-60 min x Black',
                    'Within 30-60 min x Asian', 'Within 30-60 min x Hispanic',
                    '% Black in ZIP x Black', '% Black in ZIP x Asian','% Black in ZIP x Hispanic',
                    '% Hispanic in ZIP x Black', '% Hispanic in ZIP x Asian', '% Hispanic in ZIP x Hispanic',
                    '% Asian in ZIP x Black','% Asian in ZIP x Asian','% Asian in ZIP x Hispanic',
                    '% White in ZIP x Black','% White in ZIP x Asian','% White in ZIP x Hispanic',
                    'Age x Black', 'Age x Asian', 'Age x Hispanic',
                    'Income x Black', 'Income x Asian', 'Income x Hispanic',
                    'Suburban', 'Urban', 'Within 15 Min of Hospice',
                    'Within 15-30 Min of Hospice','Within 30-60 Min of Hospice',
                    # 'More than 60 Min from Hospice',
                    '% Black in ZIP', '% White in ZIP', '% Hispanic in ZIP', '% Asian in ZIP', 'Mean Income by ZIP']

    # available_vars = [var for var in custom_order if var in R['vari_cleaned'].values]
    positions = [R[R['vari_cleaned'] == var].index[0] for var in custom_order]

    p= 0.1
    R['alpha'] = (R.p_val < 0.05) * (1-p) + p

    plt.hlines(R.index, R.low_b, R.up_b, alpha = R.alpha, color='gray', zorder=1)

    plt.vlines(0,0-1,len(R), linestyle='--', color = 'black', zorder=0)
    plt.scatter(R.coef, R.index, alpha = R.alpha, color=region_color, zorder=2)


    plt.yticks(R.index, R.vari_cleaned, fontsize=8)
    plt.gca().set_axisbelow(True)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

    plt.text(
        0.95, -1,
        "Darker indicates p-value < 0.05",
        ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes
    )
    plt.title(f'{title} Logistic Regression Coefficients with 95% CI', fontsize=8)
    plt.xlabel("Odds Ratio", fontsize=10)
    # plt.ylabel("Predictors", fontsize=10)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(left=0.4)
    # plt.tight_layout(pad=2.0)
    # plt.subplots_adjust(top=0.88)



    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_logistic_coefs.png", dpi=300)
    plt.close()

    marginal_df = marginal_effects.summary_frame()

    print(marginal_df.columns)

    marginal_df = marginal_df.reset_index().rename(columns={
        'index': 'variable',
        'dy/dx': 'marginal_effect',
        'Std. Err.': 'std_err',
        'Conf. Int. Low': 'ci_lower',
        'Cont. Int. Hi.': 'ci_upper',
        'Pr(>|z|)': 'p_value'
    })
    print(marginal_df.columns)


    # plt.figure(figsize=(8,6))
    # sns.set(style='whitegrid')


    #
    # sns.pointplot(
    #     data=marginal_df,
    #     x='marginal_effect',
    #     y='variable',
    #     join=False,
    #     ci=None,
    #     color='blue'
    # )
    # print(marginal_df.head())
    marginal_df['variable'] = marginal_df['variable'].replace(rename_dict)
    print(marginal_df['variable'])

    for idx, row in marginal_df.iterrows():
        alpha_val = 1.0 if row['p_value'] < 0.05 else 0.3
        # dot_color = 'tab:blue' if row['p_value'] < 0.05 else 'lightgray'

        plt.errorbar(
            x=row['marginal_effect'],
            y=row['variable'],
            xerr=[[row['marginal_effect'] - row['ci_lower']],
                  [row['ci_upper'] - row['marginal_effect']]],
            fmt='none', ecolor='gray', elinewidth=1, alpha=alpha_val, capsize=3, zorder=1
        )
        plt.scatter(
            row['marginal_effect'],
            row['variable'],
            color=region_color,
            alpha=alpha_val,
            label=None,
            zorder=2
        )
    plt.axvline(0, linestyle='--', color='black', linewidth=1, zorder=0)
    # plt.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)

    plt.gca().set_axisbelow(True)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

    plt.xlabel("Marginal Effect (Percentage Points)", fontsize=8)
    # plt.ylabel("Predictors", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.subplots_adjust(left=0.4)

    plt.title(f"Average Marginal Effects on Hospice Use Probability in {title}", fontsize=8)
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_hospice_margeff_coefs.png", dpi=300)
    plt.close()

    return logit_results.summary()


def fit_logistic_regression_service2(state_df, title, state, region_color):

    rename_dict = {
        'const': "Intercept",
        'is_female': 'Sex (Female)',
        'agedgrp5': 'Age Group',
        'hsa': 'Hospital Service Areas',
        'Black_pct': '% Black in ZIP',
        'White_pct': '% White in ZIP',
        'Hispanic_pct': '% Hispanic in ZIP',
        'Asian_pct': '% Asian in ZIP',
        'is15min': 'Within 15 Min of Hospice',
        'is30min': 'Within 15-30 Min of Hospice',
        'is60min': 'Within 30-60 Min of Hospice',
        # 'morethan60': 'More than 60 Min from Hospice',
        'Income': 'Mean Income by ZIP'
    }

    x_cols = ['Income', 'Hispanic_pct', 'Asian_pct', 'Black_pct', 'White_pct', 'is15min', 'is30min', 'is60min', 'Suburban',
              'Urban',
              'agedgrp5',
              'is_female']
    y_col = 'hospice_use'

    X = state_df[x_cols]
    y = state_df[y_col]

    X = sm.add_constant(X)

    X = X.astype(float)
    print(X.isnull().sum())

    logit_model = sm.Logit(y, X)
    logit_results = logit_model.fit(cov_type='cluster',
                                    cov_kwds={'groups': state_df['hsa']})

    print(logit_results.summary())

    marginal_effects = logit_results.get_margeff(method='dydx', at='overall')
    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_patient_logistic.txt', 'w') as f:
        f.write(logit_results.summary().as_text())
        f.write(marginal_effects.summary().as_text())


    R = logit_results.summary().tables[1].data
    R = pd.DataFrame(R)
    R.columns = R.iloc[0]

    R = R.iloc[1:]
    R[R.columns[1:]] = R[R.columns[1:]].astype(float)
    R.columns = ['vari', 'coef', 'std_err', 't', 'p_val', 'low_b', 'up_b']
    R['vari_cleaned'] = R['vari'].replace(rename_dict)
    # R = R.sort_values('coef', ascending=True)
    R = R.reset_index(drop=True)
    # print(R['vari_cleaned'].values)

    custom_order = ["Intercept", 'Sex (Female)',
                    'Age Group',

    'Suburban', 'Urban','Within 15 Min of Hospice',
                    'Within 15-30 Min of Hospice','Within 30-60 Min of Hospice',
                    # 'More than 60 Min from Hospice',
                    '% Black in ZIP', '% White in ZIP', '% Hispanic in ZIP', '% Asian in ZIP', 'Mean Income by ZIP']

    # available_vars = [var for var in custom_order if var in R['vari_cleaned'].values]
    positions = [R[R['vari_cleaned'] == var].index[0] for var in custom_order]

    p= 0.1
    R['alpha'] = (R.p_val < 0.05) * (1-p) + p

    plt.hlines(R.index, R.low_b, R.up_b, alpha = R.alpha, color='gray', zorder=1)

    plt.vlines(0,0-1,len(R), linestyle='--', color = 'black', zorder=0)
    plt.scatter(R.coef, R.index, alpha = R.alpha, color=region_color, zorder=2)


    plt.yticks(R.index, R.vari_cleaned, fontsize=8)
    plt.gca().set_axisbelow(True)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

    plt.text(
        0.95, -1,
        "Darker indicates p-value < 0.05",
        ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes
    )
    plt.title(f'{title} Logistic Regression Coefficients with 95% CI', fontsize=8)
    plt.xlabel("Odds Ratio", fontsize=10)
    # plt.ylabel("Predictors", fontsize=10)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(left=0.4)




    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_logistic_coefs.png", dpi=300)
    plt.close()

    marginal_df = marginal_effects.summary_frame()

    print(marginal_df.columns)

    marginal_df = marginal_df.reset_index().rename(columns={
        'index': 'variable',
        'dy/dx': 'marginal_effect',
        'Std. Err.': 'std_err',
        'Conf. Int. Low': 'ci_lower',
        'Cont. Int. Hi.': 'ci_upper',
        'Pr(>|z|)': 'p_value'
    })
    print(marginal_df.columns)



    marginal_df['variable'] = marginal_df['variable'].replace(rename_dict)
    print(marginal_df['variable'])

    for idx, row in marginal_df.iterrows():
        alpha_val = 1.0 if row['p_value'] < 0.05 else 0.3

        plt.errorbar(
            x=row['marginal_effect'],
            y=row['variable'],
            xerr=[[row['marginal_effect'] - row['ci_lower']],
                  [row['ci_upper'] - row['marginal_effect']]],
            fmt='none', ecolor='gray', elinewidth=1, alpha=alpha_val, capsize=3, zorder=1
        )
        plt.scatter(
            row['marginal_effect'],
            row['variable'],
            color=region_color,
            alpha=alpha_val,
            label=None,
            zorder=2
        )
    plt.axvline(0, linestyle='--', color='black', linewidth=1, zorder=0)


    plt.gca().set_axisbelow(True)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

    plt.xlabel("Marginal Effect (Percentage Points)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.subplots_adjust(left=0.4)

    plt.title(f"Average Marginal Effects on Hospice Use Probability in {title}", fontsize=8)
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_hospice_margeff_coefs.png", dpi=300)
    plt.close()

    return logit_results.summary()

def fit_logistic_regression_service3(state_df, title, state, region_color):

    rename_dict = {
        'const': "Intercept",
        'is_female': 'Sex (Female)',
        # 'agedgrp5_2': '70-74 Age Group',
        # 'agedgrp5_3': '75-79 Age Group',
        # 'agedgrp5_4': '80-84 Age Group',
        # 'agedgrp5_5': '85+ Age Group',
        'agedgrp5': 'Age Group',
        'is_black': 'Black',
        'is_asian': 'Asian',
        'is_hispanic': 'Hispanic',
        # 'sex_black': 'Sex x Black',
        # 'sex_asian': 'Sex x Asian',
        # 'sex_hispanic': 'Sex x Hispanic',
        # 'urban_black': 'Urban x Black',
        # 'urban_asian': 'Urban x Asian',
        # 'urban_hispanic': 'Urban x Hispanic',
        # 'suburban_black': 'Suburban x Black',
        # 'suburban_asian': 'Suburban x Asian',
        # 'suburban_hispanic': 'Suburban x Hispanic',
        # 'is15min_black': 'Within 15 min x Black',
        # 'is15min_asian': 'Within 15 min x Asian',
        # 'is15min_hispanic': 'Within 15 min x Hispanic',
        # 'is30min_black': 'Within 15-30 min x Black',
        # 'is30min_asian': 'Within 15-30 min x Asian',
        # 'is30min_hispanic': 'Within 15-30 min x Hispanic',
        # 'is60min_black': 'Within 30-60 min x Black',
        # 'is60min_asian': 'Within 30-60 min x Asian',
        # 'is60min_hispanic': 'Within 30-60 min x Hispanic',
        # 'Black_pct_black': '% Black in ZIP x Black',
        # 'Black_pct_asian': '% Black in ZIP x Asian',
        # 'Black_pct_hispanic': '% Black in ZIP x Hispanic',
        # 'Hispanic_pct_black': '% Hispanic in ZIP x Black',
        # 'Hispanic_pct_asian': '% Hispanic in ZIP x Asian',
        # 'Hispanic_pct_hispanic': '% Hispanic in ZIP x Hispanic',
        # 'Asian_pct_black': '% Asian in ZIP x Black',
        # 'Asian_pct_asian': '% Asian in ZIP x Asian',
        # 'Asian_pct_hispanic': '% Asian in ZIP x Hispanic',
        # 'White_pct_black': '% White in ZIP x Black',
        # 'White_pct_asian': '% White in ZIP x Asian',
        # 'White_pct_hispanic': '% White in ZIP x Hispanic',
        #
        # 'age_black': 'Age x Black',
        # 'age_asian': 'Age x Asian',
        # 'age_hispanic': 'Age x Hispanic',
        # 'income_black': 'Income x Black',
        # 'income_asian': 'Income x Asian',
        # 'income_hispanic': 'Income x Hispanic',
        'hsa': 'Hospital Service Areas',
        'Black_pct': '% Black in ZIP',
        'White_pct': '% White in ZIP',
        'Hispanic_pct': '% Hispanic in ZIP',
        'Asian_pct': '% Asian in ZIP',
        'is15min': 'Within 15 Min of Hospice',
        'is30min': 'Within 15-30 Min of Hospice',
        'is60min': 'Within 30-60 Min of Hospice',
        # 'morethan60': 'More than 60 Min from Hospice',
        'Income': 'Mean Income by ZIP'
    }
    robjects.globalenv['r_df'] = pandas2ri.py2rpy(state_df)

    model_code = """
    library(lme4)
    model <- glmer('hospice_use ~ Income + Hispanic_pct + Asian_pct + Black_pct + White_pct + is15min + is30min + is60min + Suburban + Urban + is_hispanic + is_asian + is_black + agedgrp5 + is_female + (1|ZIPCODE)', 
                data=r_df, 
                family='binomial')
    summary(model)

    """
    # model = lme4('hospice_use ~ Income + Hispanic_pct + Asian_pct + Black_pct + White_pct + is15min + is30min + is60min + Suburban + Urban + is_hispanic + is_asian + is_black + agedgrp5 + is_female + (1|ZIPCODE)', data=state_df, family='binomial')
    # result = model.fit()
    result = robjects.r(model_code)
    print(result)


    return result.summary()

    # marginal_effects = logit_results.get_margeff(method='dydx', at='overall')
    # with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_patient_logistic.txt', 'w') as f:
    #     f.write(logit_results.summary().as_text())
    #     f.write(marginal_effects.summary().as_text())
    #
    #
    # R = logit_results.summary().tables[1].data
    # R = pd.DataFrame(R)
    # R.columns = R.iloc[0]
    #
    # R = R.iloc[1:]
    # R[R.columns[1:]] = R[R.columns[1:]].astype(float)
    # R.columns = ['vari', 'coef', 'std_err', 't', 'p_val', 'low_b', 'up_b']
    # R['vari_cleaned'] = R['vari'].replace(rename_dict)
    # # R = R.sort_values('coef', ascending=True)
    # R = R.reset_index(drop=True)
    # # print(R['vari_cleaned'].values)
    #
    # custom_order = ["Intercept", 'Sex (Female)',
    #                 # '70-74 Age Group', '75-79 Age Group', '80-84 Age Group', '85+ Age Group',
    #                 'Age Group',
    #                 'Black', 'Asian', 'Hispanic',
    #                 # 'Sex x Black',
    #                 # 'Sex x Asian', 'Sex x Hispanic',
    #                 # 'Age x Black',
    #                 # 'Age x Asian', 'Age x Hispanic',
    #                 # 'Income x Black',
    #                 # 'Income x Asian', 'Income x Hispanic',
    #                 # 'Urban x Black', 'Urban x Asian','Urban x Hispanic', 'Suburban x Black',
    #                 # 'Suburban x Asian', 'Suburban x Hispanic',
    #                 # 'Within 15 min x Black',
    #                 # 'Within 15 min x Asian', 'Within 15 min x Hispanic',
    #                 # 'Within 15-30 min x Black',
    #                 # 'Within 15-30 min x Asian', 'Within 15-30 min x Hispanic',
    #                 # 'Within 30-60 min x Black',
    #                 # 'Within 30-60 min x Asian', 'Within 30-60 min x Hispanic',
    #                 # '% Black in ZIP x Black', '% Black in ZIP x Asian','% Black in ZIP x Hispanic',
    #                 # '% Hispanic in ZIP x Black', '% Hispanic in ZIP x Asian', '% Hispanic in ZIP x Hispanic',
    #                 # '% Asian in ZIP x Black','% Asian in ZIP x Asian','% Asian in ZIP x Hispanic',
    #                 # '% White in ZIP x Black','% White in ZIP x Asian','% White in ZIP x Hispanic',
    #                 # 'Age x Black', 'Age x Asian', 'Age x Hispanic',
    #                 # 'Income x Black', 'Income x Asian', 'Income x Hispanic',
    #                 'Suburban', 'Urban', 'Within 15 Min of Hospice',
    #                 'Within 15-30 Min of Hospice','Within 30-60 Min of Hospice',
    #                 # 'More than 60 Min from Hospice',
    #                 '% Black in ZIP', '% White in ZIP', '% Hispanic in ZIP', '% Asian in ZIP', 'Mean Income by ZIP']
    #
    # # available_vars = [var for var in custom_order if var in R['vari_cleaned'].values]
    # positions = [R[R['vari_cleaned'] == var].index[0] for var in custom_order]
    #
    # p= 0.1
    # R['alpha'] = (R.p_val < 0.05) * (1-p) + p
    #
    # plt.hlines(R.index, R.low_b, R.up_b, alpha = R.alpha, color='gray', zorder=1)
    #
    # plt.vlines(0,0-1,len(R), linestyle='--', color = 'black', zorder=0)
    # plt.scatter(R.coef, R.index, alpha = R.alpha, color=region_color, zorder=2)
    #
    #
    # plt.yticks(R.index, R.vari_cleaned, fontsize=8)
    # plt.gca().set_axisbelow(True)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    #
    # plt.text(
    #     0.95, -1,
    #     "Darker indicates p-value < 0.05",
    #     ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes
    # )
    # plt.title(f'{title} Logistic Regression Coefficients with 95% CI', fontsize=8)
    # plt.xlabel("Odds Ratio", fontsize=10)
    # # plt.ylabel("Predictors", fontsize=10)
    # plt.xticks(fontsize=8)
    # plt.subplots_adjust(left=0.4)
    # # plt.tight_layout(pad=2.0)
    # # plt.subplots_adjust(top=0.88)
    #
    #
    #
    # plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/sm_{state}_logistic_coefs.png", dpi=300)
    # plt.close()
    #
    # marginal_df = marginal_effects.summary_frame()
    #
    # print(marginal_df.columns)
    #
    # marginal_df = marginal_df.reset_index().rename(columns={
    #     'index': 'variable',
    #     'dy/dx': 'marginal_effect',
    #     'Std. Err.': 'std_err',
    #     'Conf. Int. Low': 'ci_lower',
    #     'Cont. Int. Hi.': 'ci_upper',
    #     'Pr(>|z|)': 'p_value'
    # })
    # print(marginal_df.columns)
    #
    #
    # # plt.figure(figsize=(8,6))
    # # sns.set(style='whitegrid')
    #
    #
    # #
    # # sns.pointplot(
    # #     data=marginal_df,
    # #     x='marginal_effect',
    # #     y='variable',
    # #     join=False,
    # #     ci=None,
    # #     color='blue'
    # # )
    # # print(marginal_df.head())
    # marginal_df['variable'] = marginal_df['variable'].replace(rename_dict)
    # print(marginal_df['variable'])
    #
    # for idx, row in marginal_df.iterrows():
    #     alpha_val = 1.0 if row['p_value'] < 0.05 else 0.3
    #     # dot_color = 'tab:blue' if row['p_value'] < 0.05 else 'lightgray'
    #
    #     plt.errorbar(
    #         x=row['marginal_effect'],
    #         y=row['variable'],
    #         xerr=[[row['marginal_effect'] - row['ci_lower']],
    #               [row['ci_upper'] - row['marginal_effect']]],
    #         fmt='none', ecolor='gray', elinewidth=1, alpha=alpha_val, capsize=3, zorder=1
    #     )
    #     plt.scatter(
    #         row['marginal_effect'],
    #         row['variable'],
    #         color=region_color,
    #         alpha=alpha_val,
    #         label=None,
    #         zorder=2
    #     )
    # plt.axvline(0, linestyle='--', color='black', linewidth=1, zorder=0)
    # # plt.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)
    #
    # plt.gca().set_axisbelow(True)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    #
    # plt.xlabel("Marginal Effect (Percentage Points)", fontsize=8)
    # # plt.ylabel("Predictors", fontsize=8)
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # plt.subplots_adjust(left=0.4)
    #
    # plt.title(f"Average Marginal Effects on Hospice Use Probability in {title}", fontsize=8)
    # plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_hospice_margeff_coefs.png", dpi=300)
    # plt.close()
    #
    # return logit_results.summary()

def fit_multinom_logistic_regression_service(outcome, state_df, state):
    print(f'{outcome} for {state}')

    model = smf.mnlogit(f"{outcome} ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + is15min + is30min + is60min + Income",
        data=state_df).fit()

    with open(f'../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_{outcome}_multinom.txt', 'w') as f:
        f.write(model.summary().as_text())

    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']
    errors = (conf['upper'] - conf['lower']) / 2

    coefs.plot(kind='bar', yerr=errors, figsize=(10, 6), legend=False)
    plt.title(f'{state} {outcome} Multinom Regression Coefficients with 95% CI')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/{state}_{outcome}_coefs.png", dpi=300)
    plt.close()

    return model.summary()

def refactor_df(df_patients):
    df_patients['TYPESRVC'] = df_patients['TYPESRVC'].fillna(0)
    df_patients['service_loc'] = df_patients['service_loc'].fillna(0)

    df_patients['Black_pct'] = df_patients['Black_pct'].fillna(df_patients['Black_pct'].median())
    df_patients['White_pct'] = df_patients['White_pct'].fillna(df_patients['White_pct'].median())
    df_patients['Hispanic_pct'] = df_patients['Hispanic_pct'].fillna(df_patients['Hispanic_pct'].median())
    df_patients['Asian_pct'] = df_patients['Asian_pct'].fillna(df_patients['Asian_pct'].median())
    # df_patients['TotalPop'] = df_patients['TotalPop'].fillna(df_patients['TotalPop'].median())

    # df_patients['American Indian_pct'] = df_patients['American Indian_pct'].fillna(df_patients['American Indian_pct'].median())
    df_patients['Income'] = df_patients['Income'].fillna(df_patients['Income'].median())

    # df_patients = pd.get_dummies(df_patients, columns = ['SEX', 'agedgrp5', 'RTI_RACE', 'is_covered', 'TYPESRVC', 'ruca_category'], drop_first=True)

    df_patients['hospice_use'] = df_patients['hospice_use'].astype(int)
    df_patients['service_loc'] = df_patients['service_loc'].astype(int)
    df_patients['hsa'] = df_patients['hsa'].astype(str)


    df_patients = df_patients[~df_patients['RTI_RACE'].isin([0, 3, 6])]



    race_map = {
        1: 'is_white',
        2: 'is_black',
        4: 'is_asian',
        5: 'is_hispanic'
        # ,
        # 6: 'is_native'
        # 3: 'is_other'
    }
    # for code, col_name in race_map.items():
    #     df_patients[col_name] = (df_patients['RTI_RACE'] == code).astype(int)
    #
    # df_patients = df_patients.drop('is_white', axis=1)
    df_patients['race_str'] = df_patients['RTI_RACE'].map(race_map)

    race_dummies = pd.get_dummies((df_patients['race_str']))

    race_dummies = race_dummies.drop('is_white', axis=1)
    df_patients = pd.concat([df_patients, race_dummies], axis=1)

    ruca_dummies = pd.get_dummies(df_patients['ruca_category'], drop_first=True)
    df_patients = pd.concat([df_patients, ruca_dummies], axis=1)


    # ruca_map = {"Urban": 2,
    #             "Suburban": 1,
    #             "Rural": 0}
    # df_patients['ruca_category'] = df_patients['ruca_category'].replace(ruca_map)

    dummies = pd.get_dummies(df_patients['agedgrp5'], prefix='agedgrp5', drop_first=True)
    df_patients = pd.concat([df_patients, dummies], axis=1)

    df_patients['is_female'] = (df_patients['SEX'] == 2).astype(int)

    # bins = [0, 30000, 60000, 100000, np.inf]
    # labels = ['lt30k', 'lt30k_60k', 'gt60k_100k', 'gt100k']
    #
    # df_patients['Income'] = pd.cut(df_patients['Income'], bins=bins, labels=labels)
    # df_patients['Income'] = pd.Categorical(df_patients['Income'], categories=labels, ordered=True)
    # df_patients = df_patients.dropna(subset=['Income'])
    # # df_patients.loc[df_patients['Income'].isna(), 'Income'] = 'Missing'
    #
    # print(df_patients['Income'].value_counts())
    # print(df_patients['Income'].isna().sum())
    # income_dummies = pd.get_dummies(df_patients['Income'], drop_first=True)
    # df_patients = pd.concat([df_patients, income_dummies], axis=1)
    print(df_patients.columns)
    return df_patients


def refactor_service(df_patients):
    df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].replace(90, pd.NA)
    # df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].fillna("morethan60")
    df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].fillna("greaterthan60")

    df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].replace(15, "is15min")
    df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].replace(30, "is30min")
    df_patients['MIN_ToBreak'] = df_patients['MIN_ToBreak'].replace(60, "is60min")




    service_dummies = pd.get_dummies(df_patients['MIN_ToBreak'], drop_first=True)
    df_patients = pd.concat([df_patients, service_dummies], axis=1)

    return df_patients

def region_analyze(df, region):
    print(f"Fitting model for {region}")
    # print(new_england_df.isna().sum())
    # state_results[state] = state_df
    fit_logistic_regression(df, region)
    fit_multinom_logistic_regression("hospice_category", df, region)
    fit_multinom_logistic_regression("service_loc", df, region)


    print(pd.crosstab(df['service_loc'], df['RTI_RACE']))



    # fit_multinom_logistic_regression("coverage_use", df, region)

def service_analyze(df, title, region, region_color):
    print(f"Fitting model for {title}")

    fit_logistic_regression_service1(df, title, region, region_color)
    # fit_multinom_logistic_regression_service("hospice_category", df, region)
    # fit_multinom_logistic_regression_service("coverage_use", df, region)


    # print(pd.crosstab(df['MIN_ToBreak'], df['service_loc']))

def analyze_r(df, title, region, region_color):
    print(f"Fitting model for {title}")

    fit_logistic_regression_service3(df, title, region, region_color)

def race_analyze(df, title, region, region_color):
    print(f"Fitting model for {title}")

    fit_logistic_regression_service2(df, title, region, region_color)
    # fit_multinom_logistic_regression_service("hospice_category", df, region)
    # fit_multinom_logistic_regression_service("coverage_use", df, region)


    # print(pd.crosstab(df['MIN_ToBreak'], df['service_loc']))


def interaction_vars(df_patients):
    df_patients['sex_black'] = df_patients['SEX'] * df_patients['is_black']
    df_patients['sex_asian'] = df_patients['SEX'] * df_patients['is_asian']
    df_patients['sex_hispanic'] = df_patients['SEX'] * df_patients['is_hispanic']

    df_patients['age_black'] = df_patients['agedgrp5'] * df_patients['is_black']
    df_patients['age_asian'] = df_patients['agedgrp5'] * df_patients['is_asian']
    df_patients['age_hispanic'] = df_patients['agedgrp5'] * df_patients['is_hispanic']

    df_patients['urban_black'] = df_patients['Urban'] * df_patients['is_black']
    df_patients['urban_asian'] = df_patients['Urban'] * df_patients['is_asian']
    df_patients['urban_hispanic'] = df_patients['Urban'] * df_patients['is_hispanic']

    df_patients['suburban_black'] = df_patients['Suburban'] * df_patients['is_black']
    df_patients['suburban_asian'] = df_patients['Suburban'] * df_patients['is_asian']
    df_patients['suburban_hispanic'] = df_patients['Suburban'] * df_patients['is_hispanic']


    df_patients['income_black'] = df_patients['Income'] * df_patients['is_black']
    df_patients['income_asian'] = df_patients['Income'] * df_patients['is_asian']
    df_patients['income_hispanic'] = df_patients['Income'] * df_patients['is_hispanic']

    df_patients['is15min_black'] = df_patients['is15min'] * df_patients['is_black']
    df_patients['is15min_asian'] = df_patients['is15min'] * df_patients['is_asian']
    df_patients['is15min_hispanic'] = df_patients['is15min'] * df_patients['is_hispanic']

    df_patients['is30min_black'] = df_patients['is30min'] * df_patients['is_black']
    df_patients['is30min_asian'] = df_patients['is30min'] * df_patients['is_asian']
    df_patients['is30min_hispanic'] = df_patients['is30min'] * df_patients['is_hispanic']

    df_patients['is60min_black'] = df_patients['is60min'] * df_patients['is_black']
    df_patients['is60min_asian'] = df_patients['is60min'] * df_patients['is_asian']
    df_patients['is60min_hispanic'] = df_patients['is60min'] * df_patients['is_hispanic']

    df_patients['is60min_black'] = df_patients['is60min'] * df_patients['is_black']
    df_patients['is60min_asian'] = df_patients['is60min'] * df_patients['is_asian']
    df_patients['is60min_hispanic'] = df_patients['is60min'] * df_patients['is_hispanic']

    df_patients['Asian_pct_black'] = df_patients['Asian_pct'] * df_patients['is_black']
    df_patients['Asian_pct_asian'] = df_patients['Asian_pct'] * df_patients['is_asian']
    df_patients['Asian_pct_hispanic'] = df_patients['Asian_pct'] * df_patients['is_hispanic']

    df_patients['Black_pct_black'] = df_patients['Black_pct'] * df_patients['is_black']
    df_patients['Black_pct_asian'] = df_patients['Black_pct'] * df_patients['is_asian']
    df_patients['Black_pct_hispanic'] = df_patients['Black_pct'] * df_patients['is_hispanic']

    df_patients['Hispanic_pct_black'] = df_patients['Hispanic_pct'] * df_patients['is_black']
    df_patients['Hispanic_pct_asian'] = df_patients['Hispanic_pct'] * df_patients['is_asian']
    df_patients['Hispanic_pct_hispanic'] = df_patients['Hispanic_pct'] * df_patients['is_hispanic']

    df_patients['White_pct_black'] = df_patients['White_pct'] * df_patients['is_black']
    df_patients['White_pct_asian'] = df_patients['White_pct'] * df_patients['is_asian']
    df_patients['White_pct_hispanic'] = df_patients['White_pct'] * df_patients['is_hispanic']
    return df_patients



def main():
    df_patients = getter('/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/ContextPatientUtil_All.csv')
    # print(df_patients.shape)
    #
    df_patients = df_patients[df_patients['ruca_category'].notna()]
    # print(df_patients.shape)
    #
    #
    df_patients = refactor_df(df_patients)
    df_patients = refactor_service(df_patients)
    #
    #
    # df_patients = df_patients[df_patients['hospice_category'] != 4]

    df_patients['Black_pct'] = df_patients['Black_pct'] / 10
    df_patients['White_pct'] = df_patients['White_pct'] / 10
    df_patients['Hispanic_pct'] = df_patients['Hispanic_pct'] / 10
    df_patients['Asian_pct'] = df_patients['Asian_pct'] / 10
    df_patients['American Indian_pct'] = df_patients['American Indian_pct'] / 10
    df_patients['Income'] = df_patients['Income'] / 100000

    df_patients['SEX'] = df_patients['SEX'] - 1

    df_patients = interaction_vars(df_patients)





    # print(df_patients.shape)
    #
    #
    # print(df_patients['hospice_category'].value_counts())
    # print(df_patients['coverage_use'].value_counts())
    #
    #
    # #'lt30k_60k', 'gt60k_100k', 'gt100k'
    # columns_describe = ['SEX', 'agedgrp5', 'RTI_RACE', 'STATE_x', 'hospice_use', 'hospice_category', 'service_loc', 'is_covered', 'coverage_use', 'ruca_category']
    # ct = pd.crosstab(df_patients['hospice_use'], df_patients['RTI_RACE'])
    #
    # with open('../4-Analysis/3-Analysis/Gabrielle/descriptive_stats/TransferOut/descriptive_stats.txt', 'w') as f:
    #     f.write(f'Shape: {df_patients.shape[0]} rows x {df_patients.shape[1]} columns\n\n')
    #     f.write(ct.to_string())
    #     f.write('\n\n')
    #     for col in columns_describe:
    #         f.write(f'{col} value counts:\n')
    #         f.write(df_patients[col].value_counts(dropna=False).to_string())
    #         f.write('\n\n')
    #
    #
    # # lt30k_60k + gt60k_100k + gt100k
    #
    # model = smf.logit("hospice_use ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income", data = df_patients).fit()
    # print(model.summary())
    #
    # marginal_effects = model.get_margeff(method='dydx', at='overall')
    # print("hospice_use")
    # print(marginal_effects.summary())
    # with open('../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/patient_logistic.txt', 'w') as f:
    #     f.write(model.summary().as_text())
    #     f.write(marginal_effects.summary().as_text())

    # df_patients1 = df_patients[~df_patients['coverage_use'].isin([2,4])]
    # # print(df_patients1['coverage_use'].value_counts())
    # #
    # #
    # # model = smf.logit("hospice_use ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + lt30k_60k + gt60k_100k + gt100k",
    # #     data=df_patients1).fit()
    # # print(model.summary())
    # # with open('../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/patient_coverage_use_logistic.txt', 'w') as f:
    # #     f.write(model.summary().as_text())
    # #
    # # marginal_effects = model.get_margeff(method='dydx', at='overall')
    # # print("hospice_use covered")
    # # print(marginal_effects.summary())
    # # df_patients = df_patients[~df_patients['coverage_use'].isin([2,4])]
    # # df_patients = df_patients[~df_patients['service_loc'].isin([3])]
    df_patients.to_csv('/drives/56219-Linux/56219dua/Project2018/4-Analysis/1-Data/Gabrielle/Analysis_ContextPatientUtil_test.csv')
    #
    #
    # # print(pd.crosstab(df_patients['gt100k'], df_patients['service_loc']))
    #

    # model = smf.mnlogit(
    #     "service_loc ~  SEX + ruca_category + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income",
    #     data=df_patients).fit()
    # print(model.summary())
    # marginal_effects = model.get_margeff(method='dydx', at='overall')
    # print("service_loc")
    # print(marginal_effects.summary())
    # with open('../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/patient_service_loc_multinom.txt', 'w') as f:
    #     f.write(model.summary().as_text())
    #     f.write(marginal_effects.summary().as_text())
    #
    #
    # model = smf.mnlogit(
    #     "hospice_category ~ SEX + Suburban + Urban + agedgrp5_2 + agedgrp5_3 + agedgrp5_4 + agedgrp5_5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct + Asian_pct + Income",
    #     data=df_patients).fit()
    # print(model.summary())
    #
    # marginal_effects = model.get_margeff(method='dydx', at='overall')
    # print("hospice_category")
    # print(marginal_effects.summary())
    #
    # with open('../4-Analysis/3-Analysis/Gabrielle/stats_models/TransferOut/patient_hospice_category_multinom.txt', 'w') as f:
    #     f.write(model.summary().as_text())
    #     f.write(marginal_effects.summary().as_text())

#Thesis analysis
    # df_patients = df_patients[df_patients['RTI_RACE'] == 4]
    # df_patients = df_patients[df_patients['is30min'] == True]

    #
    # df_white = df_patients[df_patients['RTI_RACE'] == 1]
    # # df_white = df_white.sample(n=1000)
    # df_black = df_patients[df_patients['RTI_RACE'] == 2]
    # # df_black = df_black.sample(n=1000)
    #
    # df_asian = df_patients[df_patients['RTI_RACE'] == 4]
    # # df_asian = df_asian.sample(n=1000)
    #
    # df_hisp = df_patients[df_patients['RTI_RACE'] == 5]
    # # df_hisp = df_hisp.sample(n=1000)N



    #
    #
    #
    # northeast = ["CT", "ME", "MA", "NH", "RI", "VT", "DE", "DC", "MD", "PA", "VA", "WV", 'NY', 'NJ']
    # northeast_df = df_patients[df_patients['STATE_x'].isin(northeast)]
    # # northeast_df = refactor_service(northeast_df)
    # print(northeast_df.columns)
    #
    #
    # new_england = ["CT", "ME", "MA", "NH", "RI", "VT"]
    # new_england_df = northeast_df[northeast_df['STATE_x'].isin(new_england)]
    # # new_england_df = new_england_df[new_england_df['RTI_RACE'] != 6]
    # # print(new_england_df.isna().sum())
    #
    #
    # mid_atlantic = ["DE", "DC", "MD", "PA", "VA", "WV"]
    # mid_atlantic_df = northeast_df[northeast_df['STATE_x'].isin(mid_atlantic)]
    # # mid_atlantic_df = mid_atlantic_df[mid_atlantic_df['RTI_RACE'] != 6]
    # print(mid_atlantic_df.isna().sum())
    #
    #
    # region2 = ['NY', 'NJ', 'PR', 'VI']
    # region2_df = northeast_df[northeast_df['STATE_x'].isin(region2)]
    #
    #
    #
    # us_df = df_patients.copy()
    # us_df['ZIPCODE'] = us_df['ZIPCODE'].astype('category')

# End thesis analysis

    #
    # #
    # # state_list = ["CT", "ME", "MA", "NH", "RI", "VT", "DE", "DC", "MD", "PA", "VA", "WV", "NJ", "NY"]
    # # state_dfs = {}
    # #
    # # for state in state_list:
    # #     state_dfs[state] = northeast_df[northeast_df['STATE_x'] == state]
    # #     service_analyze(state_dfs[state], state)
    #
    # # south = ['AL', 'FL', 'GA', 'KY', 'MI', 'SC', 'NC', 'TN']
    # # south_df = df_patients[df_patients['STATE_x'].isin(south)]
    # #
    # # midwest = ['IL', 'IN', 'MI', 'MN', 'OH', 'WI']
    # # midwest_df = df_patients[df_patients['STATE_x'].isin(midwest)]
    # #
    # # southcentral = ['AR', 'LA', 'NM', 'OK', 'TX']
    # # southcentral_df = df_patients[df_patients['STATE_x'].isin(southcentral)]
    # #
    # # # received a ConvergenceWarning
    # # central = ['IA', 'KS', 'MO', 'NE']
    # # central_df = df_patients[df_patients['STATE_x'].isin(central)]
    # #
    # # # received a ConvergenceWarning
    # # mountain = ['CO', 'MT', 'ND', 'SD', 'UT', 'WY']
    # # mountain_df = df_patients[df_patients['STATE_x'].isin(mountain)]
    # #
    # # pacific = ['AZ', 'CA', 'HI', 'NV']
    # # pacific_df = df_patients[df_patients['STATE_x'].isin(pacific)]
    # #
    # # pnw = ['AK', 'ID', 'OR', 'WA']
    # # pnw_df = df_patients[df_patients['STATE_x'].isin(pnw)]


# thesis analysis

    # service_analyze(northeast_df, "Northeast", 'northeast','blue')
    # #
    # service_analyze(new_england_df, "Region 1 (New England)", "newengland",'darkgreen')
    # service_analyze(mid_atlantic_df, "Region 3 (Mid-Atlantic)", 'midatlantic', 'lightcoral')
    # service_analyze(region2_df, "Region 2 (NY and NJ)", 'njny', 'mediumpurple')

    # analyze_r(us_df, "United States", 'us', 'blue')
    # race_analyze(df_white, "White Beneficiaries", 'white', 'red')
    # race_analyze(df_black, "Black Beneficiaries", 'black', 'green')
    # race_analyze(df_asian, "Asian Beneficiaries", 'asian', 'orange')
    # race_analyze(df_hisp, "Hispanic Beneficiaries", 'hisp', 'skyblue')
    #

#
#
#
#
# # thesis analysis end
#
#
#     # # region_analyze(south_df, "South")
#     # # region_analyze(midwest_df, "Midwest")
#     # # region_analyze(southcentral_df, "SouthCentral")
#     # # region_analyze(central_df, "Central")
#     # # region_analyze(mountain_df, "Mountain")
#     # # region_analyze(pacific_df, "Pacific")
#     # # region_analyze(pnw_df, "PNW")
#     #
#     regions_df = [us_df, northeast_df, new_england_df, mid_atlantic_df, region2_df, df_white, df_black, df_asian, df_hisp]
#     region_names = ['us', 'northeast', 'new_england', 'mid_atlantic', 'region2', 'white', 'black', 'asian', 'hisp']
#     # regions_df = [northeast_df, new_england_df, mid_atlantic_df, region2_df]
#     # region_names = ['northeast', 'new_england', 'mid_atlantic', 'region2']
#
#     # regions_df = [us_df, northeast_df]
#     # region_names = ['us', 'northeast']
#     columns_describe_service = ['SEX', 'agedgrp5', 'RTI_RACE', 'STATE_x', 'hospice_use', 'hospice_category', 'service_loc', 'is_covered', 'ruca_category', 'is15min', 'is30min', 'is60min']
#
#
#
#
#     # regions_df = [new_england_df, mid_atlantic_df, region2_df, south_df, southcentral_df, central_df, midwest_df, mountain_df, pacific_df, pnw_df]
#     # region_names = ['new_england', 'mid_atlantic', 'region2', 'south', 'southcentral', 'central', 'midwest', 'mountain', 'pacific', 'pnw']
#
#     for name, region_df in zip(region_names, regions_df):
#         ct = pd.crosstab(region_df['hospice_use'], region_df['RTI_RACE'])
#         ct2 = pd.crosstab(region_df['hospice_use'], region_df['SEX'])
#         ct3 = pd.crosstab(region_df['hospice_use'], region_df['MIN_ToBreak'])
#         ct4 = pd.crosstab(region_df['hospice_use'], region_df['ruca_category'])
#         ct5 = pd.crosstab(region_df['hospice_use'], region_df['agedgrp5'])
#
#
#
#
#
#         with open(f'../4-Analysis/3-Analysis/Gabrielle/descriptive_stats/TransferOut/{name}_descriptive_stats.txt', 'w') as f:
#             f.write(f'Shape: {region_df.shape[0]} rows x {region_df.shape[1]} columns\n\n')
#             f.write(ct.to_string())
#             f.write('\n\n')
#             f.write(ct2.to_string())
#             f.write('\n\n')
#             f.write(ct3.to_string())
#             f.write('\n\n')
#             f.write(ct4.to_string())
#             f.write('\n\n')
#             f.write(ct5.to_string())
#             f.write('\n\n')
#
#             for col in columns_describe_service:
#                 f.write(f'{col} value counts:\n')
#                 f.write(region_df[col].value_counts(dropna=False).to_string())
#                 f.write('\n\n')
#
#     # for name, region_df in zip(state_list, state_dfs):
#     #     with open(f'../4-Analysis/3-Analysis/Gabrielle/descriptive_stats/TransferOut/{name}_descriptive_stats.txt', 'w') as f:
#     #         f.write(f'Shape: {region_df.shape[0]} rows x {region_df.shape[1]} columns\n\n')
#     #         for col in columns_describe_service:
#     #             f.write(f'{col} value counts:\n')
#     #             f.write(region_df[col].value_counts(dropna=False).to_string())
#     #             f.write('\n\n')
#
#     #
#     # race_names = ['Black', 'White', 'Hispanic']
#     # for name in race_names:
#     #     race_df = northeast_df[northeast_df[f'{name}_pct'] >= 20]
#     #     print(f'Regression for {name}')
#     #     fit_logistic_regression_service(race_df, name)
#     #     # fit_logistic_regression_race(race_df, name)
#     #     # fit_multinom_logistic_regression("service_loc", race_df, name)
#     #     with open(f'../4-Analysis/3-Analysis/Gabrielle/descriptive_stats/TransferOut/{name}_descriptive_stats.txt', 'w') as f:
#     #         f.write(f'Shape: {race_df.shape[0]} rows x {race_df.shape[1]} columns\n\n')
#     #         for col in columns_describe_service:
#     #             f.write(f'{col} value counts:\n')
#     #             f.write(race_df[col].value_counts(dropna=False).to_string())1
#     #             f.write('\n\n')
#
#


if __name__ == '__main__':
    main()


