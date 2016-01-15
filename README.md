# Creating the Ideal Field Office Strategy for the Presidential Election
#### An analysis of the 2012 presidential election ground game with an eye towards 2016

### TL;DR
The goal of this research is to analyze the effect of field office placement in the 2012 presidential election.  In agreement with literature, I find that field office placement was responsible for a Democratic victory in Florida.  For the first time, I show an effective ground game for Republicans is the path to victory in Florida and potentially New Hampshire.  I also show smaller states with sharply partisan counties are more susceptible to an effective ground game and strategic field office placement.

### Introduction
The folk wisdom of the 2008 and 2012 presidential elections is that the Democrat's superior 'ground game' handed them the crucial swing states needed for an electoral victory.  By mobilizing 786 field offices <a href=http://www.mischiefsoffaction.com/2013/01/the-ground-game-in-2012.html>[1]</a> located in key counties, grassroots organizers were able to get out the vote and just edge Obama over the top.  But how much truth is there to this traditional narrative?

In a previous analysis by John Sides and Lynn Vavrek <a href=http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/>[2]</a>, field offices were found to have had a strong role in winning Florida for the Democrats. Field office placement along with other factors such as advertising spend helped Democrats to gain 248,000 votes nationally.  Similarly Joshua Darr, Luke Keele, and Matthew Levendusky <a href=http://apr.sagepub.com/content/42/3/529>[3]</a> have estimated that Obama's 2008 field operation won him about 275,000 votes and Seth Masket <a href=http://poq.oxfordjournals.org/content/73/5/1023.full>[4]</a> found ground game influential in North Carolina, Florida, and Indiana.


These analyses have typically relied on factors like median income, unemployment, demographic factors and historical voting patterns [<a href=http://apr.sagepub.com/content/42/3/529>3</a>, <a href=http://www.acweinschenk.net/uploads/1/7/3/6/17361647/psq12213.pdf>5</a>] to regress the share of the two party vote each candidate receives.  Two party vote share is the single party vote as a percentage of the total votes cast in the state for Republicans and Democrats.

While similar features will be used in my approach, I regress instead the single party vote as a percentage of voting age population.  Recent analysis <a href=http://static1.1.sqspcdn.com/static/f/864938/25350207/1408716516677/Allen+et+al.+APSA+2014.pdf?token=8SQWbYH63TzRnaVXi9JXF1mWlTY%3D>[6]</a> has shown that field office placement has the main effect of causing party aligned voters to actually vote on election day.  Ground game does not flip voters.  Thus, typical analysis of two party vote share, which depends on both the number of Democrats and Republicans, might miss part of the effect.  Instead, regressing single party vote as a percentage of voting age population removes the dependency on the other party's vote and allows us to analyze each strategy in a vacuum.

I find that counties with a historical voting bias have a larger effect from locating a field office there.  I present a strategy to maximize this effect while balancing state effects. In addition I show that only a few states are susceptible to changing their electoral votes keeping all else constant.

### Methodology
In order to understand and quantify the effect of a field office being placed in a county, I gathered data from a large number of sources to fit a linear model.  Coefficients from that linear model can estimate the effect of a field office on the single party vote.  The exogenous variables besides the field office placement bear out a detailed explanation and some visualizations of exploratory data analysis.

#### Demographic Factors
Two education factors were used: <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a> the percentage of adults without a high school diploma and the percentage of those with a bachelor's degree or higher.  In addition the county unemployment rate <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a>  was also used.

From <a href=http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp>[8]</a> nearly 15,000 columns on the rate of religious denomination participation within county populations, I used non-negative matrix factorization to reduce these columns to two latent 'topics'.  These 'topics' basically subdivide counties into two groups.  If there are different concentrations of Democratic and Republican leaning counties in these topics, then this feature will allow us to tell Democratic and Republican counties apart.

In fact, by looking at just one of the NMF features a good separation between red and blue counties is apparent as shown by the below figure.  The blue counties are much further concentrated to the left side of the chart.  Many more of the blue counties are concentrated at low values of this feature while Republican counties are concentrated at larger values.  Here Target refers to voting Democratic or Republican.

<div>
    <a href="https://plot.ly/~SGShuman/67/" target="_blank" title="Religion NMF Feature by County Winners" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/67.png" alt="Religion NMF Feature by County Winners" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:67"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 1: Religion NMF Feature Histogram 
*Histogram of the religion NMF feature split by which party the county voted for.  Democratic counties are concentrated towards the left while Republican counties have a larger mean and range.*

I did a similar analysis to the religion NMF with 16,000 more typical demographic factors taken from the US Census.  Using data points like population of a certain age, race, and residential factors, an NMF based purely on this has very strong predictive power (R2 of ~.9 without other factors).  However, it is collinear with other demographic factors and so does not appear in the final analysis.

#### County Factors
The Rural-urban Continuum Code <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a> offers insight on the nature of population density inside a county.  Low scores show localized high population density and high scores have homogeneous lower population density.

#### Historical Factors
The Cook Partisan Voting Index <a href=https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index>[10]</a> provides detail on the historical voting bias of a particular county.  The larger the magnitude of the Cook Index, the more unequal the vote in the last two presidential elections have been.  Though this is typically calculated at the state or congressional district level, here I calculated at the census county level to match the rest of the analysis.  


<div>
    <a href="https://plot.ly/~SGShuman/69/" target="_blank" title="Historical Democratic Bias by County Winners" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/69.png" alt="Historical Democratic Bias by County Winners" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:69"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 2: Cook Partisan Voting Index Feature Histogram 
*Histogram of the Cook Index feature split by which party the county voted for.  81 historically Republican counties voted Democrat while only 5 historically Democratic counties voted Republican.*

Interestingly the Cook Partisan Voting Index is not entirely predictive of the results, some counties with negative (Republican) Cook Indexes were won by Democrats. It also provides a look at which party has more counties on the extreme and overall.

The change in two party vote share (explained above) from 2008 to 2012 [<a href=https://github.com/huffpostdata/election-2012-results>11</a>, <a href=https://www.udel.edu/johnmack/frec682/red_state_blue_state/>12</a>] also gives the model the ability to predict the trend from year to year.  This feature takes the two party vote share results from 2008 and contrasts them with 2012.

<div>
    <a href="https://plot.ly/~SGShuman/73/" target="_blank" title="Change in Two Party Vote Share by County Winners" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/73.png" alt="Change in Two Party Vote Share by County Winners" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:73"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 3: Change in Two Party Vote Share Feature Histogram 
*Histogram of the change in two party vote share feature split by which party the county voted for.  There was a large regression to the mean for 414 Democratic Counties, less than 250 Republican Counties became less extreme.*

#### Campaign Expenditures and Donations
Campaign expenditures <a href=http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012>[13]</a> in a county summed over all monthly finance reports available offers some predictive power. Donations <a href=https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/>[14]</a> were investigated but not included in the final model.

#### State Level Effects - Explain how this is different from the other effects
To counter state level fixed effects, I used the 2012 state level partisan turnout <a href=http://bipartisanpolicy.org/library/2012-voter-turnout/>[9]</a> as a feature as well.  This allows the model to account for state by state differences. 

#### Office Data and Interaction
Locations of each of the 786 Obama offices and 284 Romney offices <a href=http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/>[2]</a> were obtained from John Sides who generously sent me data.  Offices have two layers of importance: their quantity, and their interaction with the Cook Index.  

Obama placed anywhere from 1 to 21 offices in a county (with by far most having one office), so I used the presence of 1 office or 2+ offices as categorical features.  I also interacted (multiplying) the Cook Index with a boolean variable indicating the presence of any number of offices.  This means that the number of votes expected from placing a field office in a county will depend on both the number of offices placed and the Cook Index.  In the end, counties with larger Cook Indexes (indicating more partisan historical voting) had a larger effect from office placement.

I won't include a map here (I'm sure you can picture a map of the US in your head), but please refer to <a href='http://blogs.lse.ac.uk/usappblog/2014/05/12/when-placed-strategically-campaign-field-offices-can-be-very-important-in-turning-battleground-states-during-presidential-elections/'> Joshua Darr's and Matthew Levendusky's maps </a> for additional clarity <a href=http://apr.sagepub.com/content/42/3/529>[3]</a> on office placement.  Cook Indexes also follow the typical pattern of showing population centers as blue and rural counties as red <a href=https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index>[10]</a>.

### Linear Model Results
As found in previous papers, I established Democratic and Republican field offices offer starkly different results.  While a field office placed by Democrats offers the chance for an increase in votes, Republican field offices have no statistical effect.  Essentially, the Republican field office strategy had no effect on turnout while Obama's offices in my model were responsible for about 2.0M votes.

Although these numbers are significantly higher than what other researchers found, the interaction term in the model changes the magnitude of the effect, but not its importance to the overall election.

There is one regression result per party, each using similar factors.  The state average turnout refers to the single party turnout and so is different for Republicans and Democrats.

##### Democratic Regression  
*Adjusted R2 - .978*

| Feature                               | Coeff      | Std Err  | P Val |
|---------------------------------------|------------|----------|-------|
| State Average Turnout                 | 0.0048     | 0.000    | 0.000 |
| Rural-urban Continuum Code            | 0.0046     | 0.000    | 0.000 |
| Perc Less than Highschool Diploma     | 0.0006     | 0.000    | 0.000 |
| Perc with Bachelor's                  | 0.0032     | 8.54e-05 | 0.000 |
| Unemployment Rate                     | 0.0041     | 0.000    | 0.000 |
| Religion NMF Feature 1                | 0.0091     | 0.001    | 0.000 |
| Religion NMF Feature 2                | 0.0063     | 0.000    | 0.000 |
| Campaign Expenditure                  | -4.084e-10 | 5.09e-11 | 0.000 |
| Cook Index                            | 0.4752     | 0.006    | 0.000 |
| Change in Vote Share 2008->2012       | 0.2689     | 0.024    | 0.000 |
| 1 Field Office                        | 0.0088     | 0.002    | 0.000 |
| 2+ Field Offices                      | 0.0257     | 0.004    | 0.000 |
| Field Office - Cook Index Interaction | 0.0348     | 0.017    | 0.041 |

###### Figure 4: Democratic Turnout Regression Results
*The factors and coefficients from the linear model.  All coefficients are significant.  Because the data are not normalized, the coefficients are not directly comparable.  In addition, to understand the effect of a field office in terms of votes added, more computation is needed.  See Simulation in Action*

Republicans with a similar R2 to Democrats show a strong fit but not a strong statistical certainty for field office placement having an effect.

##### Republican Regression  
*Adjusted R2 - .982*

| Feature                               | Coeff      | Std Err  | P Val |
|---------------------------------------|------------|----------|-------|
| State Average Turnout                 | 0.0060     | 0.000    | 0.000 |
| Rural-urban Continuum Code            | 0.0044     | 0.000    | 0.000 |
| Perc Less than Highschool Diploma     | -0.0024    | 0.000    | 0.000 |
| Perc with Bachelor's                  | 0.0025     | 0.000    | 0.000 |
| Unemployment Rate                     | 0.0054     | 0.000    | 0.000 |
| Religion NMF Feature 1                | 0.0003     | 0.001    | 0.700 |
| Religion NMF Feature 2                | 0.0072     | 0.001    | 0.000 |
| Campaign Expenditure                  | 4.905e-10  | 6.44e-11 | 0.000 |
| Cook Index                            | -0.5827    | 0.008    | 0.000 |
| Change in Vote Share 2008->2012       | -0.0543    | 0.032    | 0.000 |
| 1 Field Office                        | 0.0087     | 0.004    | 0.035 |
| 2+ Field Offices                      | 0.0143     | 0.008    | 0.080 |
| Field Office - Cook Index Interaction | -0.1054    | 0.029    | 0.000 |

###### Figure 5: Republican Turnout Regression Results
*Not all features are significant in this model, most importantly locating 2+ field offices was not significant.*

From these results, we can find the number of votes added to each county by placing a field office there, as explained in the simulation section.

#### State by State Results - might one more graph showing the difference between bars
We can now calculate the average expected number of votes to be gained by adding field offices to every county in a state.  This is seen below with the error bars indicating the maximum and minimum possible effects by state.

<div>
    <a href="https://plot.ly/~SGShuman/53/" target="_blank" title="Average Percent Vote Increase by State" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/53.png" alt="Average Percent Vote Increase by State" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:53"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 6: Average Percent Vote Increase by State
*The expected change in single party turnout by state.  Error bars indicate a maximum and minimum.  States like Arizona and NJ have a roughly equal effect, while some states like Wyoming and Alabama are very one sided.*

Our goal in the simulation is to identify states where the projected difference in votes is less than the difference between the right hand and left hand sides of the above graph.  In fact, since we can expect some statistical variation, the simulation can identify all the states who could possibly flip in any situation.  This means including simulations where one party has no ground game and the other party puts offices in every county.

### Simulation and Strategy

With the results of the linear model able to offer a statistical range of effects that are possible by placing a field office, I simulated the 2012 election 1000 times to see if any states could be flipped by placing field offices.  The goal of this simulation is to see if any statistical fluctuations can produce different results in the 2012 election.

In this simulation, I assumed that Republican's field office placement was statistically significant.  I make this assumption because in 2016 it is possible that lessons learned from the 2012 campaign could cause equal effect to Democratic strategy.  My findings do not show any causality behind the ineffective Republican operation, it could be due to bad strategy or a difference in voter habits.

#### Simulation in Action

1. Pick one party.
2. Determine the predicted votes without placing any field offices using the linear model coefficients but setting all office coefficients to zero.
3. Place 800 offices in counties around the country.  Place more offices in states with closely contested elections.  Place either one or two offices, drawn randomly. - talk about where the random comes in.
4. Draw from a normal distribution to determine the effect of a field office in every county that got one or two offices.
5. Calculate the votes added to each county and sum.
6. Repeat for the other political party.
7. Compare results.  Calculate state winners.
8. Repeat 1000 times.

##### Calculation Details:  
* office_effect is drawn from a normal distribution with a mean of office_coef and std of office_stderr  
* interaction_effect is drawn from similar normal distribution multiplied by County Cook Index  
* Votes Added = Voting Age Population x (office_effect + interaction_effect)

```python
office_effect = np.random.normal(loc=office_coef, scale=office_stderr)
interaction_effect = Cook_Index['county_name'] * np.random.normal(loc=interaction_coef, 
                                                                  scale=interaction_stderr)
Votes Added = CVAP * (office_effect + interaction_effect)
```

The below graph shows the battleground states under traditional understanding.  The highlighted bubbles indicate which states have single party vote percentages for each party within 5% of each other.  The size indicates the number of electoral votes.  This means the states that we would consider battlegrounds create the highlighted group below.

<div>
    <a href="https://plot.ly/~SGShuman/55/" target="_blank" title="Swing States - by Close Votes" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/55.png" alt="Swing States - by Close Votes" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:55"  src="https://plot.ly/embed.js" async></script>
</div>
###### Figure 7: Swing States with the Traditional View
*Highlighted states reveal the typical battleground states.  The size of the bubble is the number of electoral votes.  The color reveals the winner in the 2012 election.  Click to enable hover.*

However, applying what we learned about the possible range of the field office effect in each state, we find that only the highlighted states below could possibly switch.  Conventional Wisdom on swing states is upheld but limited.

<div>
    <a href="https://plot.ly/~SGShuman/57/" target="_blank" title="Swing States - Simulation Results" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/57.png" alt="Swing States - Simulation Results" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:57"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 8: Swing States according to the Simulation
*Only Ohio, North Carolina, Florida and New Hampshire remain as battleground states.*

#### Simulation Results and Insights
My model predicts every state correctly (Florida by only 10,000 votes) and indeed shows Obama flipping Florida with his effective field office organization.  This means that Obama loses Florida if neither party mounts a ground game, and wins if Democrats do have ground game (remember Republican ground game was ineffective).  The other insight my model provides into Florida is that Democrats only win Florida in one out of every one thousand simulations when the Republicans mount a strong field organization there.  That is, if Republicans can match the Democrats effect, they take Florida nearly every time.  Obama did indeed flip Florida in 2012 but the Republicans could easily have answered with an effective field game of their own.

Another big surprise is New Hampshire, where small changes can have a larger effect.  For every thousand times the simulation is run, New Hampshire will switch to the Republicans 100-250 times, while Florida, Ohio, and North Carolina will only switch 5-10 out of every 1000 times.

So, what is going on?  Why is New Hampshire, with all 4 of its electoral votes so susceptible to field office placement strategy.

First let's discuss what two factors are important in adding votes: the county size, and the office effect.  The county size is the most important factor, the larger a county is, the more votes are added to the state due to field office placement.  If that county contains a large percentage of the state population, then this can dominate the states total added votes.  The other factor is the office effect which is the magnitude of the expected increase in votes drawn from the normal distributions defined above.  County size does not change simulation to simulation, but the size of office effect does because of its random nature.

So, in order to flip a state, one county must contain an appreciable percentage of the total state voting population and there must be a strong office effect in this particular simulation.  In addition the state must have few counties so the office effect is not negated by random fluctuations in other counties.

The visualization below shows this.  The size of the bubble is the size of the office effect unscaled while the height is the percentage of the population that lives in that county.  Counties with higher positions are more influential in terms of population and larger bubbles also indicates larger average office effect.

<div>
    <a href="https://plot.ly/~SGShuman/71/" target="_blank" title="Influential Counties by State" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/71.png" alt="Influential Counties by State" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:71"  src="https://plot.ly/embed.js" async></script>
</div>

###### Figure 9: Largest Counties in Several Important states
*The top ten largest counties in several states are shown.  The height is the influence due to population and the size is the size of the office effect (closely tied to Cook Index) for the losing party.  Notice that Los Angeles is too tiny to see since Republicans will not ever raise many votes there with a ground game.  Coloring is the winner of the county in the 2012 election.  New Hampshire clearly shows a pattern other states don't.*


Let's go over a scenario in New Hampshire where Republicans might win.  First the Republicans play and see a strong office effect in Rockingham and Hillsborough Counties (Rockingham is much more likely to have a large office effect due to its Cook Index).  The Democrats then play and see small office effects in Hillsborough and Rockingham counties.  So, New Hampshire swings red due to statistical fluctuations, even with a smart ground game strategy on both sides.  In California, this isn't possible despite Los Angeles being so important in terms of population.


#### The Ideal Strategy
This simulation shows that field office placement could play a vital role in Republican strategy in 2012. With a more effective strategy getting out the vote in Florida, its 29 electoral votes will be collected by the Republicans.  With a long shot in NH (especially against Bernie Sanders), field offices can optimally change the single party turnout in a state by up to 5-7%.

In many of the swing states, if each party mounts an effective ground game, then nothing is gained or lost.  In Ohio, where there was a 3% difference in vote share in 2012 in favor of the Dems, an average ground game gives a 3% shift in favor of Obama.  In states where the margin is less close, the ground game matters even less.

However, this assumes that both Democrats and Republicans blanket every county with field offices.  What these findings suggest is a more subtle strategy.  Closer to election day, the probable county results will be known with much more precision.  By choosing states that are very close near election day, the field office strategy can have a large effect.  By placing offices in historically biased counties with large populations, each party can limit the resources spent on each state while maximizing votes gained.  For Democrats this can essentially mean target cities while for Republicans it may suggest a more local approach of finding strongly partisan pockets in densely populated areas.

Overall given limited resources, only states that are statistically close enough to be turned should be considered.  Given the bar chart above, we can now find those.

### Next Steps
The unit of analysis for this research was the county level.  However, as counties are contiguous, it is not at all clear that this is an appropriate unit of analysis (although it is a convenient one).  Next steps could focus on both a more granular approach and the causes or predictors or ineffectiveness in a field office.  Clues as to what could make the Republican ground game more effective could be very important.

The simulation also assumed an even strategy between Democrats and Republicans.  In other simulations, even giving Dems no offices and Republicans thousands, we see similar results.  Not all states can be flipped.  Further analysis should focus on this asymmetry to find more innovative strategies for each party.

#### References
[1] *The Ground Game in 2012*, Seth Masket. Mischiefs of Faction  
http://www.mischiefsoffaction.com/2013/01/the-ground-game-in-2012.html

[2] *How Much Did the 2012 Air War and Ground Game Matter?*, John Sides. The Monkey Cage  
http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/

[3] *Relying on the Ground Game: The Placement and Effect of Campaign Field Offices*, Joshua P. Darr, Matthew S. Levendusky. American Politics Research May **2014** vol. 42 no. 3 529-548  
http://apr.sagepub.com/content/42/3/529

[4] *Did Obama's Ground Game Matter?*, Seth E. Masket. Public Opin Q (**2009**) 73 (5): 1023-1039.  
http://poq.oxfordjournals.org/content/73/5/1023.full

[5] *Campaign Field Offices and Voter Mobilization in 2012*, Aaron C. Weinschenk. Presidential Studies Quarterly 45, no. 3 (September)  
http://www.acweinschenk.net/uploads/1/7/3/6/17361647/psq12213.pdf

[6] *Office Space: A Geo-Spatial Analysis of the Effects of Field Offices on Voter Turnout*, Seth Masket et al. American Political Science Association, August 28-31, **2014**, Washington, D.C.  
http://static1.1.sqspcdn.com/static/f/864938/25350207/1408716516677/Allen+et+al.+APSA+2014.pdf?token=8SQWbYH63TzRnaVXi9JXF1mWlTY%3D

[7] United States Department of Agriculture Economic Research Service  
http://www.ers.usda.gov/data-products/county-level-data-sets.aspx

[8] Association of Relgion Data Archives  
http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp

[9] Bipartisan Policy Center, 2012 Voter Turnout Report.  
http://bipartisanpolicy.org/library/2012-voter-turnout/

[10] Cook Partisan Voting Index (Self Calculated)  
https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index

[11] HuffPostData Election 2012 Results, Aaron Bycoffe, Jay Boice.  
https://github.com/huffpostdata/election-2012-results

[12] Homework Data Set, John Mackenzie.  
https://www.udel.edu/johnmack/frec682/red_state_blue_state/

[13] Federal Election Commission: Detailed Files About Candidates, Parties and Other Committees  
http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012

[14] *Political influence by county: A new way to look at campaign finance data*, Ryan Sibley, Bob Lannon, Ben Chartoff. Sunlight Foundation Blog  
https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/