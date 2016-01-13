import pandas as pd
import numpy as np
import cPickle as pkl
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import os
import matplotlib.pyplot as plt
import us

from load_data import get_st, get_county, get_2008_election_results, get_2012_election_results

class Featurize(object):
	'''Feature engineering made easy'''

	def __init__(self):
		pass

	def load_summary_cols(self, path='data/summary_census'):
		'''Join all the tables from a subfolder, remove excess columns'''
		files = os.listdir(path)
		# Initialize the DataFrame
		init = files.pop()
		df = pd.read_csv(path + '/' + init, low_memory=False)
		df.drop('Unnamed: 0', axis=1, inplace=True)

		# Append unique columns to the inital data frame
		for f in files:
			temp = pd.read_csv(path + '/' + f, low_memory=False)
			temp.drop(['Unnamed: 0', 'NAME', 'state', 'county'], 
				      axis=1, inplace=True)
			df = pd.concat([df, temp], axis=1, join_axes=[df.index])

		df['state_full'] = df['NAME'].apply(self._strip_state)
		df['state_abbr'] = df['state_full'].apply(self._get_abbr)
		df['county_full'] = df['NAME'].apply(self._strip_county)
		df['county_id'] = df['state_abbr'] + '_' + df['county_full']

		df = df[df['state_abbr'] != 'AK']
		df = df[df['state_abbr'] != 'SD']
		df = df[df['state_abbr'] != 'PR']

		# Fill na with 0
		return df.fillna(0)

	def _strip_state(self, county):
		'''Return the state from after the comma'''
		return county.split(',')[1].strip()

	def _get_abbr(self, state):
	    '''Return the abbreviation of a state'''
	    return str(us.states.lookup(unicode(state)).abbr)

	def _strip_county(self, count):
		'''Return the county from county, state'''
		count = count.split()
		# Most counties end in County,
		if 'County,' in count:
			idx = count.index('County,')
		# Some end in City,
		elif 'City,' in count:
			idx = count.index('City,') + 1
		elif 'city,' in count:
			idx = count.index('city,') + 1
		# Some don't end in either, so this assumes the state is one word
		else: idx = len(count) -2
		# Take the words from name that are probably county related
		count = count[:idx]
		output = ''
		for x in count:
			output = output + x.strip('.,') + ' '
		return output.strip()

	def load_CVAP(self, fname='data/County_VAP.csv'):
		'''CVAP = Current Voting Age Population'''
		df = pd.read_csv(fname)
		df['INCITS'] = df['GEOID'].apply(lambda x: x[7:])
		df['state'] = df['GEONAME'].apply(self._strip_state)
		df['state_abbr'] = df['state'].apply(self._get_abbr)
		return df

	def load_religion(self, path='data/religion.DTA', k=5):
		'''Return nmf features from a STATA file'''
		# http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp
		df = pd.read_stata(path)
		id_df = df[['stcode', 'cntycode']].copy()
		id_df.columns = ['st_num', 'county_num']
		cols = [x for x in df.columns if 'rate' in x] #only take percentage cols
		nmf_data = df[cols].fillna(0)
		model = NMF(n_components=k).fit(nmf_data)
		features = model.transform(nmf_data)
		nmf_feats = pd.DataFrame(features)
		# Name columns for interperatibility
		nmf_feats.columns = ['relig_nmf_feat_' + str(x) for x in list(nmf_feats.columns)]
		# Join the NMF. k = number of topics / cols to add
		output = id_df.join(nmf_feats)

		return output

	def load_demo(self, path='data/summary_census', profile_path='data/profile_census/census_data_profile.csv', k=5):
		'''Return nmf features from demographic census data, both Summary and Profile'''
		df = self.load_summary_cols(path)
		df = df.join(pd.read_csv(profile_path).drop(['NAME', 'state', 'county'], axis=1))
		id_df = df[['state', 'county']].copy()
		id_df.columns = ['st_num', 'county_num']
		df = df.drop(['state', 'county'], axis=1)

		# Only take columns that are integers
		cols = df.columns[df.dtypes != object]
		nmf_data = df[cols].fillna(0)
		nmf_data = normalize(nmf_data, norm='l1', axis=0) # preprocessing
		model = NMF(n_components=k).fit(nmf_data)
		features = model.transform(nmf_data)
		nmf_feats = pd.DataFrame(features)
		# Name columns for interperatibility
		nmf_feats.columns = ['demo_nmf_feat_' + str(x) for x in list(nmf_feats.columns)]
		# Join the NMF. k = number of topics / cols to add
		output = id_df.join(nmf_feats)

		return output

	def load_education(self, fname='data/county_data/Education.xls'):
		'''Return a DataFrame of education features'''
		# http://www.ers.usda.gov/data-products/county-level-data-sets.aspx
		df = pd.read_excel(fname, header=2)
		df.drop(0, inplace=True) # Drop the summary row
		# Get FIPS code in the same format as other documents
		df['FIPS Code'] = df['FIPS Code'].astype(float).astype(str) 
		df['st_num'] = df['FIPS Code'].apply(get_st)
		df['county_num'] = df['FIPS Code'].apply(get_county)
		return df

	def load_unemployment(self, fname='data/county_data/Unemployment.xls'):
		'''Return a DataFrame of unemployment features'''
		df = pd.read_excel(fname, header=6)
		# Get FIPS code in the same format as other documents
		df['FIPS_Code'] = df['FIPS_Code'].astype(float).astype(str) 
		df['st_num'] = df['FIPS_Code'].apply(get_st)
		df['county_num'] = df['FIPS_Code'].apply(get_county)
		return df

	def load_poverty(self, fname='data/county_data/PovertyEstimates.xls'):
		''' Return a DataFrame of poverty features'''
		df = pd.read_excel(fname, header=2)
		df.drop(0, inplace=True) # Drop the summary row
		# Get FIPS code in the same format as other documents
		df = df[df['Area_Name'] != 'Kalawao County'] # No data
		df['FIPStxt'] = df['FIPStxt'].astype(float).astype(str) 
		df['st_num'] = df['FIPStxt'].apply(get_st)
		df['county_num'] = df['FIPStxt'].apply(get_county)
		return df

	def normalize_by_col(self, df, cols, col='st_num'):
		'''Normalize one col by unique values in aonther column'''
		for feat in cols:
			df[feat] = pd.to_numeric(df[feat], errors='coerce')
		grouped = df.groupby(col).max().fillna(0)
		grouped[col] = grouped.index.copy()
		df_merged = pd.merge(df, grouped, on=col, suffixes=('','_st_avg'))
		for feat in cols:
			if df[feat].dtype != 'object':
				df[feat] = df[feat] / df_merged[feat + '_st_avg']
		return df

	def load_turnout(self, fname, prefix='tot', columns=None):
		'''Load data on turnout by county

		Taken from a pdf so the columns were switched
		Data was manually cleaned
		'''
		col_list = ['2012_p_vote',
		            '2008_p_vote',
		            '2004_p_vote',
		            '2000_p_vote',
		            '1996_p_vote',
		            '2012_VAP',
		            '2012_turnout',
		            '2008_delta',
		            '1992_p_vote',
		            '2004_delta',
		            '2000_delta',
		            '1996_delta',
		            '1992_delta']

		df = pd.read_csv(fname, ' ', header=None)
		if columns:
			df.columns = columns
		else:
			col_list = [prefix + '_' + x for x in col_list]
			df.columns = ['state_abbr'] + col_list

		return df

	def load_offices(self, fname, suffix='dem'):
		'''Load data on organizing offices by county'''
		df = pd.read_csv(fname, "|")
		df['county_stripped'] = df['County'].apply(lambda x: x.strip('"').replace('.', ''))
		
		df['county_id'] = df['State'] + '_' + df['county_stripped']
		output = df[['county_id','County']].groupby('county_id').count()
		output.columns = ['Num_offices_' + suffix]
		return output

	def get_electoral_df(self, path='data/electoral_college.csv'):
		'''Return a DataFrame with the electoral college inside'''
		df = pd.read_csv(path, '\t', header=None)
		df.columns = ['state_abbr', 'electoral_votes']
		return df

	def load_donations(self, path='data/political_influence_by_county_20131023.csv'):
		'''Return a DataFrame of donations by county'''
		#https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/
		df = pd.read_csv(path)
		df = df[df['cycle'] == 2012]
		df = df[['state_code', 'county_code', 'dem_prez_amount', 'rep_prez_amount', 'diff_rep-dem_percent', 'est_population']]
		df['dem_prez_amount'] /= df['est_population']
		df['rep_prez_amount'] /= df['est_population']
		df.drop('est_population', axis=1, inplace=True)
		df.columns = ['st_num', 'county_num', 'dem_prez_amount', 'rep_prez_amount', 'Rep Cont % - Dem Cont %']
		return df

	def load_expenditures(self, path='data/expenditures_cleaned.csv'):
		'''Return a DataFrame of Expenditures by county'''
		# http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012
		df = pd.read_csv(path)
		df = df.groupby(['STATE'], as_index=False).sum()
		# df['county_id'] = df['STATE'] + '_' + df['county']
		df = df[['STATE', 'TRANSACTION_AMT']].copy()
		df.columns = ['state_abbr', 'expenditure']
		return df

	def calc_cook(self, path12='data/election_2012_cleaned.csv', path08='data/raw_data/vote08_by_county.xls'):
		'''https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index

		Calculate the cook score based on the description in the link
		Typically at state or congressional district level, here calculated for county'''
		df_08 = get_2008_election_results(path08)
		df_12 = pd.read_csv(path12)

		df_08['vote_share_08'] = df_08['OBAMA'] / (df_08['OBAMA'] + df_08['MCCAIN']).astype(float)
		
		dem_mask = df_12['dem'] == 1
		rep_mask = df_12['rep'] == 1
		
		# Get Democratic Votes
		df_merged = pd.merge(df_08, df_12[dem_mask][['votes', 'st_num', 'county_num']], on=['st_num', 'county_num'], how='left')
		new_cols = list(df_merged.columns)
		new_cols[-1] = 'votes_dem_12'
		df_merged.columns = new_cols

		# Get Republican Votes
		df_merged = pd.merge(df_merged, df_12[rep_mask][['votes', 'st_num', 'county_num']], on=['st_num', 'county_num'], how='left')
		new_cols = list(df_merged.columns)
		new_cols[-1] = 'votes_rep_12'
		df_merged.columns = new_cols

		df_merged['vote_share_12'] = df_merged['votes_dem_12'] / (df_merged['votes_dem_12'] + df_merged['votes_rep_12']).astype(float)

		av_08 = df_08['OBAMA'].sum() / float(df_08['OBAMA'].sum() + df_08['MCCAIN'].sum())
		av_12 = df_12[dem_mask]['votes'].sum() / float(df_12[dem_mask]['votes'].sum() + df_12[rep_mask]['votes'].sum())

		av_overall = (av_08 + av_12) / 2.

		df_merged['avg_vote_share'] = (df_merged['vote_share_08'] + df_merged['vote_share_12']) / 2.
		df_merged['cook_score'] = df_merged['avg_vote_share'] - av_overall
		df_merged['delta_vote_share'] = df_merged['vote_share_12'] - df_merged['vote_share_08']

		# More positive means more democratic bias
		return df_merged[['st_num', 'county_num', 'cook_score', 'delta_vote_share', 'vote_share_12']]

