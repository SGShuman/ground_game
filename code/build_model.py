# coding=utf-8

import numpy as np
import pandas as pd
from code.load_data import Census_Data_Loader
from code.featurize import Featurize
import os
from sklearn.cross_validation import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statsmodels.stats.api as smf
import math
from scipy.stats import shapiro, kstest, anderson
from scipy.stats.mstats import normaltest

def join_cvap_census(census_data, CVAP):
    '''Return one df merged on county'''
    temp_df = CVAP[CVAP['LNTITLE']=='Total'][['GEONAME', 'CVAP_EST']]
    output = pd.merge(census_data, temp_df, left_on='NAME', right_on='GEONAME', how='left')
    # the special character breaks the join, fill it in
    output.loc[output['NAME'] == 'Doña Ana County, New Mexico', 'CVAP_EST'] = 156094 # from census
    output.drop('GEONAME', axis=1, inplace=True)
    return output.fillna(0)

def join_turnout(census_data, turnout):
    '''Return one df merged on state'''
    output = pd.merge(census_data, turnout, on='state_abbr', how='left')
    return output.fillna(0)

def join_election(census_data, election_data):
    '''Return one df merged on state and county'''
    output = pd.merge(census_data, election_data, left_on=['state', 'county'], right_on=['st_num', 'county_num'], how='left')
    output.drop(['st_num', 'county_num'], axis=1, inplace=True)
    return output

def join_expenditure(census_data, expenditure_data):
    df = pd.merge(census_data, expenditure_data, on='state_abbr', how='left')
    return df.fillna(0)

def join_offices(census_data, office_data):
    '''Return one df merged on state and county'''
    office_data['county_id'] = office_data.index.values
    df = pd.merge(census_data, office_data, on='county_id', how='left')
    return df.fillna(0)

def make_joined_df(census_data, CVAP, turnout, election_data, office_data, featurizer, mod_type, k=2):
    '''Return one df performing all of the above merges, for one party'''
    # Get columns for joining on
    temp_df = census_data[['NAME', 'state_abbr', 'state', 'county', 'county_id']]
    # Join Voting Age population
    temp_df = join_cvap_census(temp_df, CVAP)
    
    # Choose cols from turnout to include
    turnout_cols = ['2012_p_vote']
    # Suffix with dem or rep
    turnout_cols = ['state_abbr'] + [mod_type + '_' + x for x in turnout_cols]
    temp_df = join_turnout(temp_df, turnout[turnout_cols])

    # Add a minimum to this column to account for low outliers
    temp_df[mod_type + '_' + '2012_p_vote'] = temp_df[mod_type + '_' + '2012_p_vote'].apply(lambda x: max([x, 15]))
    
    # Choose cols to merge on
    identifiers = ['st_num', 'county_num']

    education = featurizer.load_education()
    # Choose cols from education info to include
    education_cols = ['2013 Rural-urban Continuum Code', 
                      'Percent of adults with less than a high school diploma, 2009-2013',
                      'Percent of adults with a bachelor\'s degree or higher, 2009-2013']
    temp_df = join_election(temp_df, education[identifiers + education_cols])
    # Is the county rural or not, 0 is metro, 1 is rural
    # temp_df['Metro Bool'] = temp_df['2013 Rural-urban Continuum Code'].apply(lambda x: x <= 3).astype(int)
    # temp_df.drop('2013 Rural-urban Continuum Code', inplace=True, axis=1)
    
    unemployment = featurizer.load_unemployment()

    # Choose unemployment columns
    unemployment_cols = ['Unemployment_rate_2013']
    temp_df = join_election(temp_df, unemployment[identifiers + unemployment_cols])
    
    # Add a demographic NMF
    # demo = featurizer.load_demo(k=k)
    # temp_df = join_election(temp_df, demo)

    # Add a religion NMF
    religion = featurizer.load_religion(k=k)
    temp_df = join_election(temp_df, religion)

    # Add a donations df, not predictive
    # donations = featurizer.load_donations()
    # temp_df = join_election(temp_df, donations)

    # Add an expenditures df
    expenditures = featurizer.load_expenditures()
    temp_df = join_expenditure(temp_df, expenditures)

    # Add a cook score and vote share change
    cook = featurizer.calc_cook()
    temp_df = join_election(temp_df, cook)

    # Join the office data
    temp_df = join_offices(temp_df, office_data)
    # No difference between 2 offices or 3, so group into 1 and 2+ groups
    temp_df['1_office'] = (temp_df['Num_offices_' + mod_type] == 1).astype(int)
    temp_df['2_office'] = (temp_df['Num_offices_' + mod_type] >= 2).astype(int)
    temp_df.drop('Num_offices_' + mod_type, axis=1, inplace=True)

    # Add interaction term
    temp_df['cook * office_bool'] = (temp_df['1_office'] + temp_df['2_office']) * temp_df['cook_score']

    # Alaska and South Dakota need to be dropped since districts don't align to census districts
    # Get only votes for the party the df represents
    mask = election_data[mod_type] == 1
    mask2 = election_data['st_num'] != 2 # Alaska
    mask3 = election_data['st_num'] != 46 # South Dakota
    temp_df = join_election(temp_df, election_data[mask & mask2 & mask3][['st_num', 'county_num', 'votes']])

    # Fill NA with 0, there aren't many
    return temp_df.fillna(0)

def make_X_y(df, mod_type='dem'):
    '''Return X and y for a statsmodels fit and feature names'''
    X = df.drop(['NAME', 'state_abbr', 'state', 'county', 'county_id', 'votes', 'vote_share_12', 'CVAP_EST'], axis=1)
    cols = X.columns
    # Number of votes divided by voting age population of the party
    perc_vote = df['votes'] / df['CVAP_EST'].astype(float)
    
    # Vote share, another option for regression
    # if mod_type == 'dem':
    #     y = df['vote_share_12']
    # else:
    #     y = 1 - df['vote_share_12']

    y = perc_vote

    return X, y.values, cols

def print_fi(model, feature_names, featurizer, num_print=10, prints_on=False, plots_on=False):
    '''Print or Plot feature importances with feature names for a GB or RF from sklearn'''
    idx_imp_feat = np.argsort(model.feature_importances_)[::-1]
    imp = model.feature_importances_ / max(model.feature_importances_)
    if prints_on:
        for x, y in zip(feature_names[idx_imp_feat[:num_print]], imp[idx_imp_feat[:num_print]]):
            print 'Feature: %s, Importance: %s' % (featurizer.get_col_desc(x), y)
    if plots_on:
        plt.bar(left=range(int(num_print)), 
                height=imp[idx_imp_feat[:num_print]])
        plt.xlabel('Features')
        plt.ylabel('Importances')
        plt.title('Feature Importances, scaled')
        plt.xticks(range(int(num_print)), feature_names[idx_imp_feat[:num_print]], rotation=90)

def get_kfold(model, X, y):
    '''Print MSE and R2 for sklearn models'''
    kf = KFold(X.shape[0], n_folds=5, shuffle=True)
    r2, mse = [], []
    for train_index, test_index in kf:
        model.fit(X[train_index], y[train_index])
        y_pred = model.predict(X[test_index])
        r2.append(r2_score(y[test_index], y_pred))
        mse.append(mean_squared_error(y[test_index], y_pred))
    
    print 'Model: %s, R2: %s, MSE: %s' % (model.__class__.__name__, np.mean(r2), np.mean(mse))

def run_counterfactual(model, df, mod_type):
    '''Print a counterfactual where no offices are present

    Input:
    model: sklearn model without parenthesis
    df: from function make_joined_df
    mod_type: dem or rep

    Output:
    Print the result of the counterfactual'''

    X, y, _ = make_X_y(df, mod_type=mod_type)
    # Fit linear model without the office data
    X_counter, _, _ = make_X_y(df.drop(['1_office', '2_office'], axis=1), mod_type=mod_type)
    lr_count = model().fit(X_counter, y)
    lr_norm = model().fit(X, y)

    # Create masks to identify counties for simulation
    has_1_offices = X['1_office'].values == 1
    has_2_offices = X['2_office'].values == 1

    # Get voter turnout in the counterfactual in counties that have offices
    votes_counter_1 = lr_count.predict(X_counter[has_1_offices])
    votes_counter_2 = lr_count.predict(X_counter[has_2_offices])

    # Get voter turnout from the office scenario in coutnies that have offices
    votes_norm_1 = lr_norm.predict(X[has_1_offices])
    votes_norm_2 = lr_norm.predict(X[has_2_offices])

    # Count votes for the factual and counterfactual
    num_votes_counter_1 = sum(votes_counter_1 * df[has_1_offices]['CVAP_EST'].values)
    num_votes_counter_2 = sum(votes_counter_2 * df[has_2_offices]['CVAP_EST'].values)

    num_votes_norm_1 = sum(votes_norm_1 * df[has_1_offices]['CVAP_EST'].values)
    num_votes_norm_2 = sum(votes_norm_2 * df[has_2_offices]['CVAP_EST'].values)

    print 'Num_Offices, Normal, Counterfactual, Difference, CVAP'
    print '1', num_votes_norm_1, num_votes_counter_1, num_votes_norm_1 - num_votes_counter_1, np.sum(df[has_1_offices]['CVAP_EST'].values)
    print '2+', num_votes_norm_2, num_votes_counter_2, num_votes_norm_2 - num_votes_counter_2, np.sum(df[has_2_offices]['CVAP_EST'].values)

    print 'Number of votes added'
    print np.round(sum([num_votes_norm_1 - num_votes_counter_1, num_votes_norm_2 - num_votes_counter_2]))

def plot_resids(fit_model, y):
    '''Plot the studentized residuals against the target'''
    s_resid = fit_model.resid / np.std(fit_model.resid)
    plt.scatter(y, s_resid, alpha=.2)
    slope, intercept = np.polyfit(y, s_resid, 1)
    plt.plot(y, np.poly1d(np.polyfit(y, s_resid, 1))(y))
    plt.xlabel('Voter Turnout')
    plt.title('Studentized Residuals')
    plt.ylabel('Residuals')
    print 'Slope = %s' % slope
    plt.show()

def plot_box_resids(fit_model, y_pred, subset=None):
    s_resid = (fit_model.resid - np.mean(fit_model.resid)) / np.var(fit_model.resid)
    if subset:
        s_resid = np.random.choice(s_resid, replace=False, size=math.floor(len(s_resid) * subset))
    df = pd.DataFrame(s_resid, columns=['resids'])
    temp_df = pd.DataFrame(y_pred, columns=['target'])
    df = df.join(temp_df)
    
    if min(y_pred) < -1:
        df['turnout_bucket'] = df['target'].apply(lambda x: int(math.floor(10 * np.exp(x))))
        y = df['target'].apply(lambda x: np.exp(x))
    else:
        df['turnout_bucket'] = df['target'].apply(lambda x: int(math.floor(10 * x)))
        y = df['target']

    posit = sorted(df['turnout_bucket'].unique())    

    plt.scatter(y, s_resid, alpha=.2)
    slope, intercept = np.polyfit(y, s_resid, 1)
    plt.plot(y, np.poly1d(np.polyfit(y, s_resid, 1))(y))
    plt.title('Studentized Residuals vs Prediction')
    plt.xlabel('Predicted Value')
    plt.ylabel('Studentized Residual')
    print 'Slope of best fit line: %s' % slope
    plt.show()
    
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1 = df[['resids', 'turnout_bucket']].boxplot(by='turnout_bucket', positions=posit, widths=.5)
    plt.title('Residuals versus Turnout')
    plt.xlabel('Turnout Bucket')
    plt.ylabel('Studentized Residuals')
    plt.suptitle('')
    plt.show()
    
    fig = sm.qqplot(s_resid, line='s')
    plt.title('Q-Q Plot')
    plt.show()
    
    w, p_val = shapiro(s_resid)
    print 'Shapiro-Wilk P_val is %s, larger the better' % p_val
    
    k, p_val = normaltest(s_resid)
    print 'D’Agostino and Pearson’s P_val is %s, larger the better' % p_val
    
    k, p_val = kstest(s_resid, 'norm')
    print 'Kolmogorov–Smirnov P_val is %s, larger the better' % p_val
    
    A, critical, sig = anderson(s_resid)
    print 'Anderson-Darling A2 is %s, smaller the better' % A
    print critical
    print sig
    
    n, bins, patches = plt.hist(s_resid, 75, normed=1)
    mu = np.mean(s_resid)
    sigma = np.std(s_resid)
    plt.plot(bins, mlab.normpdf(bins, mu, sigma))
    plt.title('Residuals versus a Normal Dist')
    plt.show()
    
    df['turnout_bucket'].hist(bins=posit, align='left', color='b')
    plt.title('Histogram of Turnout Bucket')
    plt.ylabel('Count')
    plt.xlim(-.5, - .5 + len(posit))
    
    rects = df['turnout_bucket'].hist(bins=posit, align='left').patches
    temp = df[['resids', 'turnout_bucket']].groupby('turnout_bucket').count()
    temp.columns = ['Count']
    labels = temp.values
    
#     for rect, label in zip(rects, labels):
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    plt.show()
    print temp

if __name__ == '__main__':
    featurizer = Featurize()
    print 'Loading Data...'
    # 2013 ACS summary data
    census_data = featurizer.load_summary_cols()
    # 2012 Election Data
    election_data = pd.read_csv('data/election_2012_cleaned.csv')
    election_data.drop('Unnamed: 0', axis=1, inplace=True)
    # 2013 Citizens of voting age by county
    CVAP = featurizer.load_CVAP()
    # Location of Field offices 2012
    obama_offices = featurizer.load_offices('data/Obama_Office_Locations_Parsed_Cleaned.csv', suffix='dem')
    romney_offices = featurizer.load_offices('data/Romney_Office_Locations_Parsed_Cleaned.csv', suffix='rep')
    # Turnout by state
    dem_turnout = featurizer.load_turnout('data/turnout/democratic_turnout.csv', prefix='dem')
    rep_turnout = featurizer.load_turnout('data/turnout/republican_turnout.csv', prefix='rep')

    print 'Making df and fitting NMF...'
    obama_df = make_joined_df(census_data, CVAP, dem_turnout, election_data, obama_offices, featurizer, mod_type='dem', k=2)
    romney_df = make_joined_df(census_data, CVAP, rep_turnout, election_data, romney_offices, featurizer, mod_type='rep', k=2)

    X_obama, y_obama, feat_names_obama = make_X_y(obama_df, mod_type='dem')
    X_romney, y_romney, feat_names_romney = make_X_y(romney_df, mod_type='rep')

    print 'Fitting models and printing results...'
    glsar_model = sm.GLSAR(y_obama, X_obama, rho=1)
    glsar_results = glsar_model.iterative_fit(1)
    print glsar_results.summary()

    # Plot residuals against fit target
    plot_box_resids(glsar_results, glsar_results.fittedvalues)