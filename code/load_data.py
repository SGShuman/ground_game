import numpy as np
import pandas as pd
import requests
import us  # to get state information
from bs4 import BeautifulSoup  # wikipedia tables
import os
import itertools
import geonameszip


class Census_Data_Loader(object):
	'''Load data from the American Communities Survey
	INPUT: CENSUS_API_KEY, freely available at
	http://api.census.gov/data/key_signup.html
	All loading is done at the county level
	'''

	def __init__(self, CENSUS_API_KEY, year=2013, api_type='summary'):
		self.api_key = CENSUS_API_KEY
		self.year = str(year)
		self.api_type = api_type

	def load_columns(self):
		'''Load the available columns for the census

		INPUT: year of census to investigate, int
		OUTPUT: Pandas DataFrame of columns and household type data identifier
		suffix: E is total #, PE is percent, M is margin, PM percent margin
		'''

		if self.api_type == 'summary':
			cols_link = 'http://api.census.gov/data/'+self.year+'/acs5/variables.json'
		else:
			cols_link = 'http://api.census.gov/data/' +\
				self.year + '/acs5/profile/variables.json'

		census_cols = pd.read_json(cols_link)
		census_cols['unpacked'] = census_cols['variables'].apply(unpack)
		census_cols['name'] = census_cols.index.astype(str)

		return census_cols

	def get_col_list_by_type(self, census_cols, dtype='E'):
		'''Get the column strings for a E, M, PM, or PE columns

		E is the total number, M is the Margin of Error
		P indicates percentage'''

		names = census_cols['name']
		# Choose the correct mask to only get numerical columns
		if self.api_type == 'summary':
			mask = names.apply(lambda x: x[0] in 'BC')
		else:
			mask = names.apply(lambda x: x[:2] == 'DP')
		names = names[mask]

		col_list = []
		if dtype in 'EM':
			for x in names:
				if 'P' in x[-2]:
					continue
				else:
					if x[-1] == dtype:
						col_list.append(x)
		else:
			for x in names:
				if x[-2] == dtype:
					col_list.append(x)
		return col_list

	def get_one_col(self, col_name='NAME'):
		'''Get a single column from the API'''
		if self.api_type == 'summary':
			search = 'http://api.census.gov/data/' + \
			        self.year + '/acs5?get=%s&for=county:*' % (col_name)
		else:
			search = 'http://api.census.gov/data/' + \
			        self.year+'/acs5/profile?get=%s&for=county:*' % (col_name)
		search += '&key=' + self.api_key
		df = pd.read_json(search, orient='DataFrame')
		df.columns = df.iloc[0]
		df.drop(0, inplace=True)
		return df

	def get_all_cols(self, col_list, start=0):
		'''For every 500 cols in the col_list, get those cols and save to file'''
		# Get just the names
		output = self.get_one_col()

		print 'Hitting API...'
		i = start
		for col in col_list[start:]:
			print 'Getting col: %s' % col
			temp_df = self.get_one_col(col)
			output = pd.concat([output, temp_df[col]], axis=1)
			if (i % 500 == 0) & (i > 0):
				with open('data/summary_census/summary_census_cols_%s.csv' % i, 'w') as f:
					output.to_csv(f, encoding='utf-8')
				output = self.get_one_col()
			i += 1

	def get_population(self, search_string='NAME&P0010001'):
		'''Return the population from the year evaluated'''
		search = 'http://api.census.gov/data/2010/sf1?get=%s&for=county:*' \
		        % search_string
		search += '&key=' + self.api_key
		df = pd.read_json(search, orient='DataFrame')
		df.columns = df.iloc[0]
		df.drop(0, inplace=True)
		return df


def unpack(data_dict):
	'''Return values from a specific data_dict'''
	concept = data_dict['concept']
	label = data_dict['label']
	return str(concept) + ': ' + str(label)


def get_st(fip):
	'''Return the state code from fips'''
	if fip == '0.0':
		return 2  # Alaska
	if fip == 'nan':
		return 0

	if len(fip) == 6:
		fip = '0' + fip
	return int(fip[:2].strip('.'))


def get_county(fip):
    '''Return the county code from fips'''
    if fip == '0.0':
    	return 1  # Alaska
    if fip == 'nan':
    	return 0

    if len(fip) == 6:
    	fip = '0' + fip
    return int(fip[2:5])


def get_candidate(candidate, text):
	'''Return a boolean if candidate in text'''
	if candidate.lower() in text.lower():
		return 1
	else:
		return 0


# Get 2012 election results and make a DataFrame
def get_2012_election_results(path='data/raw_data/election-2012-results/data'):
	'''Return the 2012 election_results as a DataFrame

	https://github.com/huffpostdata/election-2012-results'''
	files = os.listdir(path)
	files.remove('ma_towns.csv')
	files.remove('ri_towns.csv')
	files.remove('ak_precincts.csv')

	df = pd.DataFrame([[0, 'Alaska', 'Obama', 122640]])  # Data from wikipedia
	df = df.append([[0, 'Alaska', 'Romney', 164767]])
	df.columns = ['fips', 'county', 'candidate', 'votes']
	for filename in files:
		filename = path + '/' + filename
		df = df.append(pd.read_csv(filename))

	df['dem'] = df['candidate'].apply(lambda x: get_candidate('obama', x))
	df['rep'] = df['candidate'].apply(lambda x: get_candidate('romney', x))
	df['other'] = ((1 + df['dem'] + df['rep']) - 2) * -1
	df['fips'] = df['fips'].astype(str)

	df['st_num'] = df['fips'].apply(get_st)
	df['county_num'] = df['fips'].apply(get_county)
	df.dropna(inplace=True)
	return df


def get_2008_election_results(path='data/raw_data/vote08_by_county.xls'):
	'''Return a DataFrame contain the 08 election results

	'''
	df = pd.read_excel(path)
	df['FIPS'] = df['FIPS'].astype(float).astype(str)
	df['st_num'] = df['FIPS'].apply(get_st)
	df['county_num'] = df['FIPS'].apply(get_county)
	df.dropna(inplace=True)
	return df


def strip_period(county):
	'''Strip a period from a county

	St. Louis -> St Louis
	'''
	output = ''
	for word in county.split():
		output += word.strip('.') + ' '
	return output.strip()


def get_county_from_zip_fast(zip_code_series):
	'''Return Pandas Series of counties from Pandas Series of zip_codes'''
	zip_code_series = zip_code_series.astype(str)
	zips = zip_code_series.unique()
	zip_county_dict = {}
	its = 0
	for zip_code in zips:
		print its
		its += 1  # This can take while so you'll want to watch
		try:  # This step can take 300 ms per zip code
			zip_county_dict[zip_code] = geonameszip.\
			                            lookup_postal_code(zip_code[:5], 'US')['county']
		except:
			zip_county_dict[zip_code] = 'Not_Found'

	return zip_code_series.apply(lambda x: zip_county_dict[x])


def get_cand_from_id(_id, match, mod_type='dem'):
	'''Return a candidate from a candidate id'''
	if _id == match:
		return mod_type
	else:
		if mod_type == 'dem':
			return 'rep'
		else:
			return 'dem'


def load_expenditures(path='data/raw_data/expenditures/'):
	'''Return a DataFrame with the relevant expenditure data'''
	# http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012
	obama_id = 'P80003338'  # FEC website
	romney_id = 'P80003353'

	# Get opperational expendtirues
	expend = pd.read_csv(path + 'oppexp.txt', '|', header=None)
	expend.drop(25, axis=1, inplace=True)  # extra empty column
	# Column names
	expend_head = pd.read_csv(path + 'oppexp_header_file.csv')
	expend.columns = list(expend_head.columns)

	# Link candidates to commitees
	link = pd.read_csv(path + 'ccl.txt', '|', header=None)
	# Column names
	link_head = pd.read_csv(path + 'ccl_header_file.csv')
	link.columns = list(link_head.columns)

	# temp_df now contains candidates associated with each committee
	temp_df = expend.merge(link[['CMTE_ID', 'CAND_ID']], on='CMTE_ID', how='left')

	obama_mask = temp_df['CAND_ID'] == obama_id
	romney_mask = temp_df['CAND_ID'] == romney_id
	year_mask = temp_df['RPT_YR'] == 2012
	# Report type is montly spending
	report_type_mask = temp_df['RPT_TP'].apply(lambda x:
		               (x[0] == 'M') & (x[1] != 'Y'))

	temp_df = temp_df[(obama_mask | romney_mask) &
	                  year_mask & report_type_mask].copy()

	temp_df['candidate'] = temp_df['CAND_ID'].apply(lambda x:
		                   get_cand_from_id(x, match=obama_id, mod_type='dem'))

	print 'Doing Zip Code Lookups'
	temp_df['county'] = get_county_from_zip_fast(temp_df['ZIP_CODE'])
	temp_df['county'] = temp_df['county'].apply(strip_period)

	return temp_df


if __name__ == '__main__':
	API_KEY = os.environ['CENSUS_API']  # replace with your api key
	census = Census_Data_Loader(API_KEY, year=2013)

	# Get the columns from the summary data
	# print 'Getting cols table...'
	# census_cols = census.load_columns(api_type='summary')
	# print 'Done with that'
	# Save the table for later since we scraped it
	# with open('data/census_cols_summary.csv', 'r') as f:
	# 	census_cols = pd.read_csv(f, encoding='utf-8')

	# Get unique columns
	# print 'Getting cols...'
	# cols = census.get_col_list_by_type(census_cols)

	# Get the census_data from the summary api
	# print 'On to census data...'
	# census_data = census.get_all_cols(cols, 10001)
	# with open('data/census_data_2013_summary.csv', 'w') as f:
	# 	census_data.to_csv(f, encoding='utf-8')

	# pop = census.get_population()
	# with open('data/census_pop.csv', 'w') as f:
	# 	pop.to_csv(f, encoding='utf-8')

	# Must fork https://github.com/SGShuman/election-2012-results
	# election = get_2012_election_results()
	# with open('data/election_2012_cleaned.csv', 'w') as f:
	# 	election.to_csv(f)

	# Create Expenditures df
	expenditures = load_expenditures()
	with open('data/expenditures_cleaned.csv', 'w') as f:
		expenditures.to_csv(f, encoding='utf-8')
