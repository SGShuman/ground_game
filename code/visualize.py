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
from plotly import tools
from plotly.tools import FigureFactory as FF 


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
	net_effect = -100 * state_inc_rep['av_increase % of Predicted'] + 100 * state_inc_dem['av_increase % of Predicted']
	idxes = np.argsort(net_effect)

	tracea = go.Bar(
	    y=state_inc_rep.index[idxes],
	    x=-100 * state_inc_rep['av_increase % of Predicted'][idxes],
	    name='Republican Effect',
	    orientation='h',
	    marker=dict(
	        color=red
	    ),
	    error_x=dict(
            type='data',
            array=100 * state_inc_rep['error_x'][idxes],
            visible=True,
            thickness=1.5,
            width=3,
            opacity=.75
        )
	)
	traceb = go.Bar(
	    y=state_inc_rep.index[idxes],
	    x=100 * state_inc_dem['av_increase % of Predicted'][idxes],
	    name='Democratic Effect',
	    orientation='h',
	    marker=dict(
	        color=blue
	    ),
	    error_x=dict(
            type='data',
            array=100 * state_inc_dem['error_x'][idxes],
            visible=True,
            thickness=1.5,
            width=3,
            opacity=.75
        )
	)
	error_x = []
	for i in xrange(len(net_effect)):
		if net_effect[idxes][i] > 0:
			error_x.append((state_inc_dem['av_increase % of Predicted'][idxes][i] + 
				           state_inc_dem['error_x'][idxes][i]) -
			               (state_inc_rep['av_increase % of Predicted'][idxes][i] -
			               	state_inc_rep['error_x'][idxes][i]))
		else:
			error_x.append((state_inc_rep['av_increase % of Predicted'][idxes][i] + 
				           state_inc_rep['error_x'][idxes][i]) -
			               (state_inc_dem['av_increase % of Predicted'][idxes][i] -
			               	state_inc_dem['error_x'][idxes][i]))


	tracec = go.Bar(
	    y=state_inc_rep.index[idxes],
	    x=net_effect[idxes],
	    name='Net Average Vote Increase',
	    orientation='h',
	    showlegend=False,
	    marker=dict(
	        color=[red if x < 0 else blue for x in net_effect[idxes]]
	    ),
	     error_x=dict(
            type='data',
            array=100 * np.array(error_x),
            visible=True,
            thickness=1.5,
            width=3,
            opacity=.75
        )
	)
	data = [tracea, traceb]
	layout = go.Layout(
	    barmode='overlay',
	    title='Vote Increase by state',
	    height=1000
	)

	fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Average Percent Vote Increase', 
		                                                      'Net Percent Vote Increase'))
	fig.append_trace(tracea, 1, 1)
	fig.append_trace(traceb, 1, 1)
	fig.append_trace(tracec, 1, 2)
	fig['layout'].update(barmode='overlay', title='Vote Increase by State', height=1000, showlegend=False)
	plot_url = py.plot(fig, filename=title)

def plot_tables():
	data_matrix_dem = [
	['Feature', 'Coeff', 'Std Err', 'P Val'],
	['State Average Turnout', 0.0048, 0.000, 0.000],
	['Rural-urban Continuum Code', 0.0046, 0.000, 0.000],
	['Perc Less than Highschool Diploma', 0.0006, 0.000, 0.000],
	['Perc with Bachelor\'s', 0.0032, 8.54e-05, 0.000],
	['Unemployment Rate', 0.0041, 0.000, 0.000],
	['Religion NMF Feature 1', 0.0091, 0.001, 0.000],
	['Religion NMF Feature 2', 0.0063, 0.000, 0.000],
	['Campaign Expenditure', -4.084e-10, 5.09e-11, 0.000],
	['Cook Index', 0.4752, 0.006, 0.000],
	['Change in Vote Share 2008->2012', 0.2689, 0.024, 0.000],
	['1 Field Office', 0.0088, 0.002, 0.000],
	['2+ Field Offices', 0.0257, 0.004, 0.000],
	['Field Office - Cook Index Interaction', 0.0348, 0.017, 0.041]
	]

	data_matrix_rep =[
	['Feature', 'Coeff', 'Std Err', 'P Val'],
	['State Average Turnout', 0.0060, 0.000, 0.000],
	['Rural-urban Continuum Code', 0.0044, 0.000, 0.000],
	['Perc Less than Highschool Diploma', -0.0024, 0.000, 0.000],
	['Perc with Bachelor\'s', 0.0025, 0.000, 0.000],
	['Unemployment Rate', 0.0054, 0.000, 0.000],
	['Religion NMF Feature 1', 0.0003, 0.001, 0.700],
	['Religion NMF Feature 2', 0.0072, 0.001, 0.000],
	['Campaign Expenditure', -4.905e-10, 6.44e-11, 0.000],
	['Cook Index', -0.5827, 0.008, 0.000],
	['Change in Vote Share 2008->2012', -0.0543, 0.032, 0.000],
	['1 Field Office', 0.0087, 0.004, 0.025],
	['2+ Field Offices', 0.0143, 0.008, 0.080],
	['Field Office - Cook Index Interaction', -.1054, 0.029, 0.000]
	]

	table_dem = FF.create_table(data_matrix_dem)
	table_dem.layout.update({'title': 'Democratic Regression<br><i>Adj R2 0.978<i>'})
	table_rep = FF.create_table(data_matrix_rep)
	table_rep.layout.update({'title': 'Republican Regression<br><i>Adj R2 0.982<i>'})

	# fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Democratic Regression<br><i>Adj R2 0.978<i>', 
	#  	                                                      'Republican Regression<br><i>Adj R2 0.982<i>'))

	plot_url = py.plot(table_dem, filename='Dem Regression Results')
	plot_url = py.plot(table_rep, filename='Rep Regression Results')


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

def get_more_annotations_state(state_list):
	#['AZ', 'GA', 'NC', 'NV', 'PA', 'FL', 'OH', 'VA', 'IA', 'CO', 'NH', 'WI'])
	annotations = dict(
		AZ=dict(
			x=.2349,
			y=.2827,
			xref='x',
			yref='y',
			text='AZ',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		GA=dict(
			x=.2614,
			y=.3063,
			xref='x',
			yref='y',
			text='GA',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		NC=dict(
			x=.3146,
			y=.3279,
			xref='x',
			yref='y',
			text='NC',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		NV=dict(
			x=.2956,
			y=.2579,
			xref='x',
			yref='y',
			text='NV',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		PA=dict(
			x=.3102,
			y=.2780,
			xref='x',
			yref='y',
			text='PA',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3,
			ax=-15
			),
		FL=dict(
			x=.3156,
			y=.3101,
			xref='x',
			yref='y',
			text='FL',
			ax=-30,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		OH=dict(
			x=.3269,
			y=.3077,
			xref='x',
			yref='y',
			text='OH',
			ax=-3,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		VA=dict(
			x=.3393,
			y=.3136,
			xref='x',
			yref='y',
			text='VA',
			ax=0,
			ay=-50,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		IA=dict(
			x=.3635,
			y=.3229,
			xref='x',
			yref='y',
			text='IA',
			showarrow=True,
			arrowhead=1,
			arrowsize=.3,
			ax=-6
			),
		CO=dict(
			x=.3664,
			y=.3282,
			xref='x',
			yref='y',
			text='CO',
			ax=0,
			ay=-50,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		NH=dict(
			x=.3669,
			y=.3276,
			xref='x',
			yref='y',
			text='NH',
			ax=20,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			),
		WI=dict(
			x=.3818,
			y=.3323,
			xref='x',
			yref='y',
			text='WI',
			ax=30,
			showarrow=True,
			arrowhead=1,
			arrowsize=.3
			)
		)

	output = []
	for state in state_list:
		output.append(annotations[state])
	return output

def swing_state_bubble_plot(state_obama_df, state_romney_df,
	                        color, close_calls, voting_pop,
	                        size, text, title, more_annotations):
	'''Plot the swing state bubble plot'''
	annotations = [
	    dict(
			x=.5,
			y=.47,
			xref='x',
			yref='y',
			text='More Electoral Votes<br>Swing State',
			showarrow=False,
			align='left'
	        ),
	    dict(
			x=.5,
			y=.53,
			xref='x',
			yref='y',
			text='Fewer Electoral Votes<br>Not Swing State',
			showarrow=False,
			align='left'
	        ),

	]

	if more_annotations:
		annotations += more_annotations

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
	trace1 = go.Scatter(
		x=[.4, .4],
		y=[.53, .47],
		mode='markers',
		marker=dict(
			color=['rgb(255, 65, 54, .3)', 'rgb(93, 164, 214, 1)'],
			opacity=[.3, 1],
			size=[10, 15])
		)
	data = [trace0, trace1]
	layout = go.Layout(
	    title=title,
	    showlegend=False,
	    hovermode='closest',
	    annotations=annotations,
	    xaxis=dict(
	        range=[0, .6],
	        title='% Democratic Vote',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        range=[0, .6],
	        title='% Republican Vote',
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
	for i, x in enumerate(dem.df['state_abbr'].values):
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

	shapes = [
		# Bottom
		dict(
			type='circle',
			fillcolor=red,
			xref='paper',
			yref='paper',
			x0=.82,
			x1=.83,
			y0=.8,
			y1=.825,
			line=dict(color=red)
			),
		# Top
		dict(
			type='circle',
			fillcolor=blue,
			xref='paper',
			yref='paper',
			x0=.815,
			x1=.835,
			y0=.85,
			y1=.9,
			line=dict(color=blue)
			)

	]
	annotations = [
	    dict(
			x='NH',
			y=.29229539372092217,
			xref='x',
			yref='y',
			text='Hillsborough County',
			showarrow=True,
			arrowhead=1,
			arrowsize=.5
	        ),
	    dict(
			x='CA',
			y=.2478272,
			xref='x',
			yref='y',
			text='Los Angeles County',
			showarrow=True,
			arrowhead=1,
			arrowsize=.5,
			ax=-25
	        ),
	    dict(
			x='NH',
			y=.2246,
			xref='x',
			yref='y',
			text='Rockingham County',
			showarrow=True,
			arrowhead=1,
			arrowsize=.5,
			ax=25
	        ),
	    dict(
			x='MO',
			y=.28,
			xref='x',
			yref='y',
			text='   Larger Office Effect (dem)',
			showarrow=False,
			xanchor='left'
	        ),
	    dict(
			x='MO',
			y=.32,
			xref='x',
			yref='y',
			text='   Smaller Office Effect (rep)',
			showarrow=False,
			xanchor='left'
	        ),

	]

	return states, population_perc, colors, np.array(size_effect), text, annotations


def influ_counties_plot(states, population_perc, colors, 
	                    size_effect, text, annotations, title, red, blue):
	'''Plot the infuential counties scatter'''
	trace0 = go.Scatter(
	    x=states,
	    y=population_perc,
	    mode='markers',
	    text=text,
	    marker=dict(
	        color=colors,
	        size=np.absolute(size_effect) * 2000,
	        opacity=.8
	    )
	)
	trace1 = go.Scatter(
		x=['MO', 'MO'],
		y=[.32, .28],
		mode='markers',
		marker=dict(
			color=[red, blue],
			size=[7.5, 15])
		)
	data = [trace0, trace1]
	layout = go.Layout(
	    title=title,
	    showlegend=False,
	    hovermode='closest',
	    #shapes=shapes,
	    annotations=annotations,
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
	        range=[0, .35],
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
	# plot_feat_vs_y_true(x_dem, x_rep, 'Cook Partisan Voting Index',
	# 	                'Historical Democratic Bias by County Winners', red, blue, bins)

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
	more_annotations = get_more_annotations_state(['AZ', 'GA', 'NC', 'NV', 'PA', 'FL', 'OH', 'VA', 'IA', 'CO', 'NH', 'WI'])
	swing_state_bubble_plot(state_obama_df, state_romney_df,
		                    color, close_calls, voting_pop,
		                    size, text, 'Swing States - Traditional Understanding', more_annotations)

	# Swing State Bubble - Smart
	close_calls_smart = get_close_calls(state_obama_df, state_romney_df,
		                                state_inc_dem, state_inc_rep)
	more_annotations = get_more_annotations_state(['NC', 'FL', 'OH', 'NH'])
	swing_state_bubble_plot(state_obama_df, state_romney_df,
		                    color, close_calls_smart, voting_pop,
		                    size, text, 'Swing States from Field Office Placement', more_annotations)

	# States by County Effect
	state_list = ['NH', 'OH', 'FL', 'NC', 'VA', 'CA', 'MO', 'IN']
	states, population_perc, colors, size_effect, text, annotations =\
	inlfu_counties_vars(dem, rep, state_inc_dem,
		                obama_df, county_win_dict,
		                state_win_dict, state_list, red, blue)

	# influ_counties_plot(states, population_perc, colors,
	# 	size_effect, text, annotations, 'Influential Counties by State', red, blue)

	# plot_tables()
