import numpy as np
import pandas as pd
from code.featurize import Featurize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.stats.api as smf
from code.build_model import *
from random import random

class one_party_strat(object):
    '''The strategy for a single party during one cycle'''

    def __init__(self, df, X, y, num_offices, mod_type='dem'):
        self.df = df
        self.num_offices = num_offices
        
        # Fit a linear model to get the coefficients
        model = sm.GLSAR(y, X, rho=1).iterative_fit(1)
        [self.one_coef, self.two_coef, self.int_coef] = model.params[-3:]
        [self.one_std, self.two_std, self.int_std] = model.bse[-3:]
        
        # Return the models vote predictions (disallow negative votes)
        counter_model = sm.GLSAR(y, X.drop(['1_office', '2_office', 'cook * office_bool'], axis=1), rho=1).iterative_fit(1)
        self.df['votes_predicted'] = counter_model.predict(X.drop(['1_office', '2_office', 'cook * office_bool'], axis=1)) * self.df['CVAP_EST']
        self.df['votes_predicted'] = self.df['votes_predicted'].apply(lambda x: max(x, 0))

        # Greatly increase the speed of simulation using a dictionary
        self.data_dict = df.set_index('NAME').to_dict('index')
    
    def set_params(self, one_coef, two_coef, int_coef, one_std, two_std, int_std):
    	'''Reset the office and interaction coefficients'''
        self.one_coef, self.two_coef, self.int_coef = one_coef, two_coef, int_coef
        self.one_std, self.two_std, self.int_std = one_std, two_std, int_std
        
    def _weighted_sample(self, weights, sample_size):
    	'''Return indexes of weighted sort given weights'''
        totals = np.cumsum(weights)
        sample = []
        for i in xrange(sample_size):
            rnd = random() * totals[-1]
            idx = np.searchsorted(totals, rnd, 'right')
            sample.append(idx)
            totals[idx:] -= weights[idx]
        return sample
    
    def place_offices(self, weights):
    	'''Place 1 or 2 offices in counties randomly by county weight'''
        num_counties = len(self.data_dict)
        options = np.zeros(num_counties)
        idxes = self._weighted_sample(weights, num_counties)
        
        counter, iterator = self.num_offices, 0
        while counter > 0:
            choice = np.random.choice([1, 2]) # randomly place 1 or 2 offices
            if counter == 1: # if only one office left to place, place 1
                choice = 1
            
            # place number of offices in the next county chosen by the weighted sample
            options[idxes[iterator]] = choice
            counter += -1 * choice # decrement by num_offices placed
            iterator += 1
        self.df['num_offices_sim'] = options

        # Remake data_dict, now with offices placed
        self.data_dict = self.df.set_index('NAME').to_dict('index')
        
        # Get the vote increase for each county once offices have been places
        votes_added = []
        for county in self.data_dict.iterkeys():
            num_offices = self.data_dict[county]['num_offices_sim']
            votes_added.append(self._get_vote_increase(num_offices, county))
            
        self.df['votes_added'] = votes_added
        self.df['total_votes'] = self.df['votes_added'] + self.df['votes_predicted']
        
    def _get_vote_increase(self, num_offices, county):
    	'''Return the number of votes added to a county given a number of offices'''
        # Get office effect
        if num_offices == 1:
            office_effect =  np.random.normal(loc=self.one_coef, scale=self.one_std)
        elif num_offices == 2:
            office_effect =  np.random.normal(loc=self.two_coef, scale=self.two_std)
        else:
            return 0
            
        # Get effect of office and cook score interaction
        cook_effect = np.random.normal(loc=self.int_coef, scale=np.absolute(self.int_std))
        
        county_cook = self.data_dict[county]['cook_score']
        county_pop = self.data_dict[county]['CVAP_EST']
        
        # Typical formula for effect with interaction term
        return county_pop * (office_effect + county_cook * cook_effect)
    
    def get_av_increase(self):
    	'''Return a DataFrame with several vote metrics'''
        self.df['vote_effect'] = self.df['cook_score'] * self.int_coef + self.df['1_office'] * self.one_coef + self.df['2_office'] * self.two_coef
        self.df['max_vote_effect'] = self.df['cook_score'] * (self.int_coef + self.int_std) + self.df['1_office'] * (self.one_coef + self.one_std) + self.df['2_office'] * (self.two_coef + self.two_std)
        self.df['av_vote_increase'] = self.df['CVAP_EST'] * self.df['vote_effect'].apply(lambda x: max([0, x]))
        self.df['max_vote_increase'] = self.df['CVAP_EST'] * self.df['max_vote_effect'].apply(lambda x: max([0, x]))
        self.df['av_increase perc of Predicted'] = self.df['av_vote_increase'] / self.df['votes_predicted']
        return self.df[['NAME', 'state_abbr', 'votes_predicted', 'max_vote_increase', 'av_vote_increase', 'av_increase perc of Predicted', 'vote_effect', 'max_vote_effect', 'cook_score', 'CVAP_EST']]

class simulation(object):
    '''Given two strategy objects, run a simulation of the election'''
    
    def __init__(self, num_sims, electoral_df, dem_strat, rep_strat):
        self.num_sims = num_sims
        # Dicts makes the simulation run much faster
        self.electoral_dict = electoral_df.set_index('state_abbr').to_dict()['electoral_votes']
        self.dem_strat = dem_strat
        self.rep_strat = rep_strat
        
        # Initialize states and counties to keep track of results
        self.states = electoral_df[['state_abbr']].copy().set_index('state_abbr')
        self.states['rep'] = 0
        self.states['dem'] = 0
        
        self.counties = self.dem_strat.df[['NAME', 'state_abbr']].copy()
        self.counties['rep'] = 0
        self.counties['dem'] = 0
        
        self._calc_weights()
        
    def _calc_weights(self):
    	'''Return a list of weights of which counties to select first'''
        # Find the states that are swing states
        weights = np.absolute(self.dem_strat.df.groupby('state_abbr').sum()['votes'] -
                              self.rep_strat.df.groupby('state_abbr').sum()['votes'])
        
        # Match the state weights up with counties
        states_with_weight = self.dem_strat.df.groupby('state_abbr', as_index=False).sum()[['state_abbr']].copy()
        weights = 1. / weights # Smaller distances should have higher weights
        states_with_weight['weights'] = weights.values
        temp_df = pd.merge(self.dem_strat.df[['state_abbr']], states_with_weight, on='state_abbr')
        
        # Store matched up weights in an array
        self.weights = temp_df['weights'].values
    
    def _fit_one(self):
    	'''Place offices for a single simulation'''
        self.dem_strat.place_offices(self.weights)
        self.rep_strat.place_offices(self.weights)
    
    def _calc_elect_votes(self):
    	'''Get electoral votes for a set of office placements'''
        dem_votes = self.dem_strat.df[['state_abbr', 'total_votes']].groupby('state_abbr').sum()
        rep_votes = self.rep_strat.df[['state_abbr', 'total_votes']].groupby('state_abbr').sum()
        
        state_winner = dem_votes > rep_votes
        state_winner = state_winner.to_dict()['total_votes']
        
        dem_elect = 0
        rep_elect = 6 # start at 6 since SD + AK = 6

        for state in state_winner.iterkeys():
            if state_winner[state]:
                dem_elect += self.electoral_dict[state] # add electoral votes
                self.states['dem'][state] += 1 # store state winner
                mask = self.dem_strat.df['state_abbr'] == state
                # If state is won for the party, count those field offices as well placed
                self.counties['dem'] += self.dem_strat.df['num_offices_sim'] * mask
            else:
                rep_elect += self.electoral_dict[state]
                self.states['rep'][state] += 1
                mask = self.rep_strat.df['state_abbr'] == state
                self.counties['rep'] += self.rep_strat.df['num_offices_sim'] * mask
                
    def get_strategy(self, thresh=.45, mod_type='dem'):
    	'''Use the greedy algorithm to limited offices in counties'''
        strat  = self.counties[['NAME', 'state_abbr', mod_type]].copy()
        # mod_type column contains all places offices were placed in winning simulation by party
        weights = strat[mod_type].values

        if mod_type == 'dem':
            num_offices = self.dem_strat.num_offices
        else:
            num_offices = self.rep_strat.num_offices
        
        # Counties with the most offices placed in them
        most_placed_idxs = np.argsort(strat[mod_type].values)[::-1]
        strategy = np.zeros(len(most_placed_idxs))

        # Place 2 offices by total offices placed until the threshhold is reached, then place one office
        counter, iterator = 0, 0
        while counter < num_offices:
            idx = most_placed_idxs[iterator]
            if counter - num_offices == 1: # Only place 2 offices if there are two left
                strategy[idx] = 1
                counter += 1
            elif weights[idx] > self.num_sims * thresh: # Place 2 offices until the threshold is reached
                strategy[idx] = 2
                counter += 2
            else:
                strategy[idx] = 1
                counter += 1
            iterator += 1
        strat['strategy'] = strategy
        return strat[['NAME', 'state_abbr', 'strategy']]
    
    def run(self):
    	'''Run the simulation'''
        for i in xrange(self.num_sims):
            # print 'Simulation %s' % i
            self._fit_one()
            self._calc_elect_votes()

    def get_swing_states(self):
    	'''Return a list of the swing states'''
        self.states['swing'] = np.absolute(self.states['dem'] - self.states['rep'])
        output = self.states.drop(['AK', 'SD'], axis=0)
        return output.sort_values(by='swing')


    def plot_swing(self):
    	'''Show simulation results and plot swing states'''
    	self.states['swing'] = np.absolute(self.states['dem'] - self.states['rep'])
    	self.states['color'] = self.states['dem'] > self.states['rep']
    	self.states['color'] = self.states['color'].apply(lambda x: 'b' if x else 'r')
    	output = self.states.drop(['AK', 'SD'], axis=0)
    	print output.sort_values(by='swing').head(25)
    	sizes = (1. / output['swing']) * 1000
    	plt.scatter(output['dem'], sizes, s=100, alpha=.4, c=self.states['color'])
    	plt.title('Swing States')
    	plt.xlabel('Sims Won Dem')
    	plt.ylabel('Variability')
    	plt.show()

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

	dem = one_party_strat(obama_df, X_obama, y_obama, 800)
	rep = one_party_strat(romney_df, X_romney, y_romney, 800, mod_type='rep')

	# Activate these lines to set the republican effect equal to the democtratic effect
	model = sm.GLSAR(y_obama, X_obama, rho=1).iterative_fit(1)
	# [one_coef, two_coef, int_coef] = model.params[-3:]
	# [one_std, two_std, int_std] = model.bse[-3:]
	# rep.set_params(one_coef, two_coef, -int_coef, one_std, two_std, int_std)

	electoral = featurizer.get_electoral_df()

	print 'Running Simulation'
	sim = simulation(100, electoral, dem, rep)
	sim.run()
	sim.plot_swing()