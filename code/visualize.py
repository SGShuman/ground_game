import numpy as np
import pandas as pd
from code.featurize import Featurize
from code.simulation import *
from code.build_model import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.stats.api as smf
import plotly.plotly as py
import plotly.graph_objs as go


def get_winners(obama_df, romney_df):
	'''Return dictionaries of winner by state and county
	True means democratic win'''

	# Counties Dems won
	county_win = obama_df['votes'] > romney_df['votes']
	temp_df = obama_df[['NAME']].copy()
	temp_df['winner'] = county_win
	county_win_dict = temp_df.set_index('NAME').to_dict()['winner']

	# States Dems won
	state_win = obama_df.groupby('state_abbr').sum()['votes'] >\
	            romney_df.groupby('state_abbr').sum()['votes']
	state_win_dict = state_win.to_dict()

	return county_win_dict, state_win_dict


def feat_vs_y_true_vars(county_win_dict, df, feat):
	'''Return feature split by county winner'''
	counties = df['NAME'].values

	# Winner of each county
	mask = [county_win_dict[x] for x in counties]
	x_dem = df[mask][feat].values
	x_rep = df[np.logical_not(mask)][feat].values

	return np.array(x_dem), np.array(x_rep), feat


def plot_feat_vs_y_true(x_dem, x_rep, x_label, title, red, blue, bins):
	'''Plot Histogram of feature split by county winner'''
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
	        title='Number of Counties Won',
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
	'''Return simulation parameter DataFrames for the split bar plot'''
	state_inc_dem = dem_strat.get_av_increase().groupby('state_abbr').sum()
	state_inc_dem['av_increase % of Predicted'] = state_inc_dem['av_vote_increase'] /\
	                                              state_inc_dem['votes_predicted']
	state_inc_dem['error_x'] = (state_inc_dem['max_vote_increase'] -
		                        state_inc_dem['av_vote_increase']) /\
	                            state_inc_dem['votes_predicted']

	state_inc_rep = rep_strat.get_av_increase().groupby('state_abbr').sum()
	state_inc_rep['av_increase % of Predicted'] = state_inc_rep['av_vote_increase'] /\
	                                              state_inc_rep['votes_predicted']
	state_inc_rep['error_x'] = (state_inc_rep['max_vote_increase'] -
		                        state_inc_rep['av_vote_increase']) /\
	                            state_inc_rep['votes_predicted']

	return state_inc_dem, state_inc_rep


def split_bar_plot(state_inc_dem, state_inc_rep, title, red, blue):
	'''Plot the split bar plot from the blog post'''
	tracea = go.Bar(
	    y=state_inc_rep.index[::-1],
	    x=-100 * state_inc_rep['av_increase % of Predicted'][::-1],
	    name='Republican Effect',
	    orientation='h',
	    marker=dict(
	        color=red
	    ),
	    error_x=dict(
            type='data',
            array=100 * state_inc_rep['error_x'][::-1],
            visible=True,
            thickness=1.5,
            width=3,
            opacity=.75
        )
	)
	traceb = go.Bar(
	    y=state_inc_rep.index[::-1],
	    x=100 * state_inc_dem['av_increase % of Predicted'][::-1],
	    name='Democratic Effect',
	    orientation='h',
	    marker=dict(
	        color=blue
	    ),
	    error_x=dict(
            type='data',
            array=100 * state_inc_dem['error_x'][::-1],
            visible=True,
            thickness=1.5,
            width=3,
            opacity=.75
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


def swing_state_bubble_vars(obama_df, romney_df, electoral_dict,
	                        red, blue, thresh=.1):
    '''Return the variables needed for the swing state bubble chart'''
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
        temp = 'State: %s<br>Electoral Votes: %s<br>Voting Age Pop: %s\
               <br>Obama Vote: %s<br>Romney Vote: %s' %\
               (name, e_votes, pop, str(ob_v) + '%', str(ro_v) + '%')
        text.append(temp)
    return state_obama_df, state_romney_df, color,\
           close_calls, voting_pop, size, text


def swing_state_bubble_plot(state_obama_df, state_romney_df,
	                        color, close_calls, voting_pop,
	                        size, text, title):
	'''Plot the swing state bubble plot'''
	trace0 = go.Scatter(
	    x=state_obama_df['votes'] / voting_pop,
	    y=state_romney_df['votes'] / voting_pop,
	    mode='markers',
	    text=text,
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


def get_close_calls(state_obama_df, state_romney_df, 
	                state_inc_dem, state_inc_rep):
	'''Return states that can be swung with ground game'''
	close_calls_smart = np.zeros(state_obama_df.shape[0])
	winner = state_obama_df['votes'] - state_romney_df['votes']
	winner = winner.apply(lambda x: x > 0)

	# If the state percentage vote difference is
	# Less than the maximum vote swing available to one party
	# Then that state is a swing state
	for i in xrange(state_obama_df.shape[0]):
	    row_dem = state_inc_dem.iloc[i]
	    row_rep = state_inc_rep.iloc[i]
	    if ((row_dem['max_vote_increase'] + row_dem['votes_predicted']) >
	    	 row_rep['votes_predicted']) & np.logical_not(winner[i]):
	        close_calls_smart[i] = 1
	    elif ((row_rep['max_vote_increase'] + row_rep['votes_predicted']) >
	    	   row_dem['votes_predicted']) & winner[i]:
	        close_calls_smart[i] = 1
	    else:
	        close_calls_smart[i] = .3

	return close_calls_smart


def inlfu_counties_vars(dem_strat, rep_strat,
	                    state_inc_dem, obama_df,
	                    county_win_dict, state_win_dict,
	                    state_list, red, blue):
	'''Return the variables needed for the influential counties plot'''
	mask = obama_df['state_abbr'].isin(state_list)
	temp_df = obama_df[mask].copy()

	df_list = []
	for x in state_list:
		mask = temp_df['state_abbr'] == x
		temp = temp_df[mask].sort_values('CVAP_EST', ascending=False).head(10)
		df_list.append(temp)

	obama_df = pd.concat(df_list)

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

	text = []
	for i in xrange(len(states)):
		row = obama_df.iloc[i]
		name = row.NAME
		cook_score = row.cook_score
		temp = '%s<br>Cook Partisan Voting Index: %s' % (name, cook_score)
		text.append(temp)

	return states, population_perc, colors, np.array(size_effect), text


def influ_counties_plot(states, population_perc, colors, 
	                    size_effect, text, title):
	'''Plot the infuential counties scatter'''
	trace0 = go.Scatter(
	    x=states,
	    y=population_perc,
	    mode='markers',
	    text=text,
	    marker=dict(
	        color=colors,
	        size=size_effect * 1000,
	        opacity=.8
	    )
	)
	data = [trace0]
	layout = go.Layout(
	    title=title,
	    xaxis=dict(
	        title='State with Top 10 Counties by Voting Age Pop',
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
	    ),
	    annotations=[
	        dict(
	            x='NH',
	            y=.29229539372092217,
	            xref='x',
	            yref='y',
	            text='Hillsborough County',
	            showarrow=True,
	            arrowhead=7,
	            ax=0,
	            ay=-40
	        )
        ]
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
	dem_office_path = 'data/Obama_Office_Locations_Parsed_Cleaned.csv'
	rep_office_path = 'data/Romney_Office_Locations_Parsed_Cleaned.csv'
	obama_offices = featurizer.load_offices(dem_office_path, suffix='dem')
	romney_offices = featurizer.load_offices(rep_office_path, suffix='rep')
	# Turnout by state
	dem_turnout_path = 'data/turnout/democratic_turnout.csv'
	rep_turnout_path = 'data/turnout/republican_turnout.csv'
	dem_turnout = featurizer.load_turnout(dem_turnout_path, prefix='dem')
	rep_turnout = featurizer.load_turnout(rep_turnout_path, prefix='rep')

	print 'Making df and fitting NMF...'
	obama_df = make_joined_df(census_data,
	                          CVAP,
	                          dem_turnout,
	                          election_data,
	                          obama_offices,
	                          featurizer, mod_type='dem', k=2)

	romney_df = make_joined_df(census_data,
	                           CVAP,
	                           rep_turnout,
	                           election_data,
	                           romney_offices,
	                           featurizer, mod_type='rep', k=2)

	X_obama, y_obama, feat_names_obama = make_X_y(obama_df, mod_type='dem')
	X_romney, y_romney, feat_names_romney = make_X_y(romney_df, mod_type='rep')

	dem = one_party_strat(obama_df, X_obama, y_obama, 800)
	rep = one_party_strat(romney_df, X_romney, y_romney, 800, mod_type='rep')

	electoral = featurizer.get_electoral_df()
	electoral_dict = electoral.set_index('state_abbr')\
	                          .to_dict()['electoral_votes']

	# print 'Running Simulation'
	# sim = simulation(1000, electoral, dem, rep)
	# sim.run()

	red = 'rgb(255, 65, 54, .6)'
	blue = 'rgb(93, 164, 214, .6)'

	county_win_dict, state_win_dict = get_winners(obama_df, romney_df)

	# By Religion NMF
	x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict,
		                                     obama_df,
		                                     'relig_nmf_feat_0')
	bins = dict(start=0, end=10, size=.25)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Religion NMF Feature',
	# 	                'Religion NMF Feature by County Winners', red, blue, bins)

	# By Cook Score
	x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict,
		                                     obama_df,
		                                     'cook_score')
	bins = dict(start=-.5, end=.5, size=.1)
	plot_feat_vs_y_true(x_dem, x_rep, 'Cook Partisan Voting Index',
		                'Historical Democratic Bias by County Winners', red, blue, bins)

	# By Democratic Expenditure
	# x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict,
	#  	                                       obama_df,
	#                                          'expenditure')
	# bins = dict(start=0, end=3500000, size=250000)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Democratic Expenditure',
	#                     'Democractic Expenditure by County Winners', red, blue, bins)

	# By Delta Vote Share
	x_dem, x_rep, feat = feat_vs_y_true_vars(county_win_dict,
	 	                                       obama_df,
	                                         'delta_vote_share')
	bins = dict(start=-.25, end=.25, size=.05)
	# plot_feat_vs_y_true(x_dem, x_rep, 'Change in Two Party Vote Share',
	#                     'Change in Two Party Vote Share by County Winners', red, blue, bins)

	# Vertical split Bar Plot
	state_inc_dem, state_inc_rep = split_bar_vars(dem, rep)
	# split_bar_plot(state_inc_dem, state_inc_rep,
	# 	           'Average Percent Vote Increase by State', red, blue)

	# Swing State Bubble - Naive
	state_obama_df, state_romney_df, color, close_calls, voting_pop, size, text =\
	swing_state_bubble_vars(obama_df, romney_df,
		                    electoral_dict, red, blue, thresh=.05)
	# swing_state_bubble_plot(state_obama_df, state_romney_df,
	# 	                    color, close_calls, voting_pop,
	# 	                    size, text, 'Swing States - by Close Votes')

	# Swing State Bubble - Smart
	close_calls_smart = get_close_calls(state_obama_df, state_romney_df,
		                                state_inc_dem, state_inc_rep)
	# swing_state_bubble_plot(state_obama_df, state_romney_df,
	# 	                    color, close_calls_smart, voting_pop,
	# 	                    size, text, 'Swing States - Simulation Results')

	# States by County Effect
	state_list = ['NH', 'OH', 'FL', 'NC', 'VA', 'CA', 'MO', 'IN']
	states, population_perc, colors, size_effect, text =\
	inlfu_counties_vars(dem, rep, state_inc_dem,
		                obama_df, county_win_dict,
		                state_win_dict, state_list, red, blue)

	# influ_counties_plot(states, population_perc, colors,
	# 	size_effect, text, 'Influential Counties by State')
