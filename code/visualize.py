import numpy as np
import pandas as pd
from code.load_data import Census_Data_Loader
from code.featurize import Featurize
from code.simulation import *
from code.build_model import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.stats.api as smf


import plotly.plotly as py
import plotly.graph_objs as go

def get_winners(obama_df, romney_df):
	'''Return dictionaries of winner by state and county_win
	True means democratic win'''
	county_win = obama_df['votes'] > romney_df['votes']
	temp_df = obama_df[['NAME']].copy()
	temp_df['winner'] = county_win
	county_win_dict = temp_df.set_index('NAME').to_dict()['winner']

	state_win = obama_df.groupby('state_abbr').sum()['votes'] > romney_df.groupby('state_abbr').sum()['votes']
	state_win_dict = state_win.to_dict()

	return county_win_dict, state_win_dict

def feat_vs_y_true_vars(county_win_dict, df, feat):
	counties = df['NAME'].values

	mask = [county_win_dict[x] for x in counties]
	x_dem = df[mask][feat].values
	x_rep = df[np.logical_not(mask)][feat].values

	return x_dem, x_rep, feat


def plot_feat_vs_y_true(x_dem, x_rep, x_label, title, red, blue, bins):

	trace1 = go.Histogram(
	    x=x_dem,
	    opacity=0.75,
	    name='Democratic Won',
	    autobinx=False,
	    xbins=bins,
	    marker=dict(
	    	color=blue
	    	),
	)
	trace2 = go.Histogram(
	    x=x_rep,
	    opacity=0.75,
	    name='Republican Won',
	    autobinx=False,
	    xbins=bins,
	    marker=dict(
	    	color=red
	    	),
	)
	data = [trace1, trace2]
	layout = go.Layout(
	    barmode='overlay',
	    title=title,
	    showlegend=True,
	    xaxis=dict(
	        title=x_label,
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        title='Count',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename=title)

def split_bar_vars(dem_strat, rep_strat):
	state_inc_dem = dem_strat.get_av_increase().groupby('state_abbr').sum()
	state_inc_dem['av_increase % of Predicted'] = state_inc_dem['av_vote_increase'] / state_inc_dem['votes_predicted']
	state_inc_rep = rep_strat.get_av_increase().groupby('state_abbr').sum()
	state_inc_rep['av_increase % of Predicted'] = state_inc_rep['av_vote_increase'] / state_inc_rep['votes_predicted']
	return state_inc_dem, state_inc_rep

def split_bar_plot(state_inc_dem, state_inc_rep, title, red, blue):
	tracea = go.Bar(
	    y=state_inc_rep.index[::-1],
	    x=-1 * state_inc_rep['av_increase % of Predicted'][::-1],
	    name='Republican Effect',
	    orientation = 'h',
	    marker = dict(
	        color = red
	    )
	)
	traceb = go.Bar(
	    y=state_inc_rep.index[::-1],
	    x=state_inc_dem['av_increase % of Predicted'][::-1],
	    name='Democratic Effect',
	    orientation = 'h',
	    marker = dict(
	        color = blue
	    )
	)
	data = [tracea, traceb]
	layout = go.Layout(
	    barmode='overlay',
	    title=title,
	    height=1000
	)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename=title)


def swing_state_bubble_vars(obama_df, romney_df, electoral_dict, red, blue, thresh=.1):
    state_obama_df = obama_df.groupby('state_abbr').sum()
    state_romney_df = romney_df.groupby('state_abbr').sum()
    
    winner = state_obama_df['votes'] - state_romney_df['votes']
    voting_pop = state_obama_df['CVAP_EST']
    
    close_calls = []
    for x in xrange(winner.shape[0]):
        if np.absolute(winner[x] / float(voting_pop[x])) < thresh:
            close_calls.append(1)
        else:
            close_calls.append(.3)
            
            
    color = []
    for x in winner:
        if x > 0:
            color.append(blue)
        else:
            color.append(red)
            
    size = []
    for x in state_obama_df.index:
        size.append(electoral_dict[x])

    text = []
    for idx in xrange(len(winner)):
        row = state_obama_df.iloc[idx]
        r_row = state_romney_df.iloc[idx]
        name = row.name
        pop = row['CVAP_EST']
        e_votes = size[idx]
        ob_v = np.round(row['votes'] / float(pop), 3) * 100
        ro_v = np.round(r_row['votes'] / float(pop), 3) * 100
        temp = 'State: %s<br>Electoral Votes: %s<br>Voting Age Pop: %s<br>Obama Vote: %s<br>Romney Vote: %s' %(name, e_votes, pop, str(ob_v)+'%', str(ro_v)+'%')
        text.append(temp)
    return state_obama_df, state_romney_df, color, close_calls, voting_pop, size, text

def swing_state_bubble_plot(state_obama_df, state_romney_df, color, close_calls, voting_pop, size, text, title):
	trace0 = go.Scatter(
	    x=state_obama_df['votes'] / voting_pop,
	    y=state_romney_df['votes'] / voting_pop,
	    mode='markers',
	    text = text,
	    marker=dict(
	        color=color,
	        opacity=close_calls,
	        size=np.log(size) * 5
	    )
	)
	data = [trace0]
	layout = go.Layout(
	    title=title,
	    xaxis=dict(
	        range=[0, .6],
	        title='% Obama Vote',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        range=[0, .6],
	        title='% Romney Vote',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    height=600,
	    width=600,
	)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename=title)

def get_close_calls(state_obama_df, state_romney_df, state_inc_dem, state_inc_rep):
	close_calls_smart = np.zeros(state_obama_df.shape[0])
	winner = state_obama_df['votes'] - state_romney_df['votes']
	winner = winner.apply(lambda x: x > 0)

	for i in xrange(state_obama_df.shape[0]):
	    row_dem = state_inc_dem.iloc[i]
	    row_rep = state_inc_rep.iloc[i]
	    if ((row_dem['max_vote_increase'] + row_dem['votes_predicted']) > row_rep['votes_predicted']) & np.logical_not(winner[i]):
	        close_calls_smart[i] = 1
	    elif ((row_rep['max_vote_increase'] + row_rep['votes_predicted']) > row_dem['votes_predicted']) & winner[i]:
	        close_calls_smart[i] = 1
	    else:
	        close_calls_smart[i] = .3

	return close_calls_smart

def inlfu_counties_vars(dem_strat, rep_strat, state_inc_dem, obama_df, county_win_dict, state_win_dict, states, red, blue):
	mask = obama_df['state_abbr'].isin(states)
	obama_df = obama_df[mask].copy()
	states = obama_df['state_abbr'].values
	state_pop = state_inc_dem['CVAP_EST'].to_dict()
	
	population_perc = []
	for i in xrange(obama_df.shape[0]):
		row = obama_df.iloc[i]
		population_perc.append(row['CVAP_EST'] / float(state_pop[row['state_abbr']]))

	colors = []
	for x in obama_df['NAME']:
		if county_win_dict[x]:
			colors.append(blue)
		else:
			colors.append(red)

	size_effect = []
	effect_dem = dem.get_av_increase()['vote_effect']
	effect_rep = rep.get_av_increase()['vote_effect']
	for i, x in enumerate(obama_df['state_abbr'].values):
		if state_win_dict[x]:
			size_effect.append(effect_rep[i])
		else:
			size_effect.append(effect_dem[i])

	# temp_df = obama_df[['state_abbr', 'NAME']].copy()
	# temp_df['population_perc'] = population_perc
	# temp_df['colors'] = colors
	# temp_df['size_effect'] = size_effect
	# state_by_county_dict = temp_df.set_index('state_abbr').to_dict('series')

	return states, population_perc, colors, size_effect


def influ_counties_plot(states, population_perc, colors, size_effect, title):
	trace0 = go.Scatter(
	    x=states,
	    y=population_perc,
	    mode='markers',
	    #text = text,
	    marker=dict(
	        color=colors,
	        size=size_effect * 10000
	    )
	)
	data = [trace0]
	layout = go.Layout(
	    title=title,
	    showlegend=True,
	    xaxis=dict(
	        title='State',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        title='Percent of State Population',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename=title)
	        

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

	# Electoral College
	electoral = featurizer.get_electoral_df()
	electoral_dict = electoral.set_index('state_abbr').to_dict()['electoral_votes']

	print 'Making df and fitting NMF...'
	obama_df = make_joined_df(census_data, CVAP, dem_turnout, election_data, obama_offices, featurizer, mod_type='dem', k=2)
	romney_df = make_joined_df(census_data, CVAP, rep_turnout, election_data, romney_offices, featurizer, mod_type='rep', k=2)

	X_obama, y_obama, feat_names_obama = make_X_y(obama_df, mod_type='dem')
	X_romney, y_romney, feat_names_romney = make_X_y(romney_df, mod_type='rep')

	dem = one_party_strat(obama_df, X_obama, y_obama, 800)
	rep = one_party_strat(romney_df, X_romney, y_romney, 800, mod_type='rep')

	# print 'Running Simulation'
	# sim = simulation(1000, electoral, dem, rep)
	# sim.run()

	red = 'rgb(255, 65, 54, .6)'
	blue = 'rgb(93, 164, 214, .6)'

	county_win_dict, state_win_dict = get_winners(obama_df, romney_df)

	# By Religion NMF
	x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict, obama_df, 'relig_nmf_feat_0')
	bins = dict(start=0, end=10, size=.25)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Religion NMF Feat', 'Religion NMF Feature vs. Target', red, blue, bins)

	# By Cook Score
	x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict, obama_df, 'cook_score')
	bins = dict(start=-.5, end=.5, size=.1)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Cook Score', 'Historical Democratic Bias vs. Target', red, blue, bins)

	# By Democratic Expenditure
	# x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict, obama_df, 'expenditure')
	# bins = dict(start=0, end=3500000, size=250000)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Democratic Expenditure', 'Democractic Expenditure vs. Target', red, blue, bins)

	# Vertical split Bar Plot
	state_inc_dem, state_inc_rep = split_bar_vars(dem, rep)
	# split_bar_plot(state_inc_dem, state_inc_rep, 'Average Percent Vote Increase by State', red, blue)

	# Swing State Bubble - Naive
	state_obama_df, state_romney_df, color, close_calls, voting_pop, size, text = swing_state_bubble_vars(obama_df, romney_df, electoral_dict, red, blue, thresh=.05)
	# swing_state_bubble_plot(state_obama_df, state_romney_df, color, close_calls, voting_pop, size, text, 'Swing States - by Close Votes')

	# Swing State Bubble - Smart
	close_calls_smart = get_close_calls(state_obama_df, state_romney_df, state_inc_dem, state_inc_rep)
	# swing_state_bubble_plot(state_obama_df, state_romney_df, color, close_calls_smart, voting_pop, size, text, 'Swing States - Simulation Results')

	# States by County Effect
	state_list = ['NH', 'OH', 'FL', 'NC', 'VA', 'CA', 'MO', 'IN']
	states, population_perc, colors, size_effect = inlfu_counties_vars(dem, rep, state_inc_dem, obama_df, county_win_dict, state_win_dict, state_list, red, blue)
	print 'Trying to Plot The Hard One'
	influ_counties_plot(states, population_perc, colors, size_effect, 'NH has one very influential county')

