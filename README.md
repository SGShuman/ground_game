# Creating the Ideal Field Office Strategy for the Presidential Election
#### An analysis of the 2012 presidential election with an eye towards 2016

### TL;DR
The goal of this research is to analyze the effect of field office placement in 2012.  In agreement with literature, I find that field office placement was responsible for a Democratic victory in Florida.  For the first time, I show an effective ground game for Republicans is the path to victory in Florida and potentially New Hampshire. Smaller states with sharply partisan counties are more susceptible to field office placement.

### Introduction
The folk wisdom of the 2008 and 2012 presidential elections is that the Democrat's superior 'ground game' handed them the crucial swing states needed for an electoral victory.  By mobilizing 786 field offices <a href=http://www.mischiefsoffaction.com/2013/01/the-ground-game-in-2012.html>[1]</a> located in key counties, grass roots supporters were able to get out the vote and just edge Obama over the top.  But how much truth is there to this traditional narrative?

In a previous analyses by John Sides and Lynn Vavrek <a href=http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/>[2]</a>, field offices were found to have had a strong role in winning Florida for the democrats and along with other factors such as advertising spend about helping to gain 248,000 votes nationally.  Similarly Joshua Darr, Luke Keele, and Matthew Levendusky <a href=http://apr.sagepub.com/content/42/3/529>[3]</a> have estimated that Obama's 2008 field operation won him about 275,000 votes and Seth Masket <a href=http://poq.oxfordjournals.org/content/73/5/1023.full>[4]</a> found ground game influential in North Carolina, Florida, and Indiana.

These analyses have typically relied on factors like median income, unemployment, demographic factors and historical voting patterns [<a href=http://apr.sagepub.com/content/42/3/529>3</a>, <a href=http://www.acweinschenk.net/uploads/1/7/3/6/17361647/psq12213.pdf>5</a>] to regress the share of the two party vote each candidate receives.  While similar features will be used in my approach, I regress instead the single party vote as a percentage of voting age population.  Recent analysis <a href=http://static1.1.sqspcdn.com/static/f/864938/25350207/1408716516677/Allen+et+al.+APSA+2014.pdf?token=8SQWbYH63TzRnaVXi9JXF1mWlTY%3D>[6]</a> has shown that field office placement has the main effect of causing party aligned voters to actually vote.  So, regressing single party turnout for each party is a meaningful target.

I find that counties with a historical voting bias are more susceptible to a large turnout effect for the historically enfranchised party and present a strategy of office placement to maximize this effect.

### Methodology
In order to understand the effect of a field office being placed in a county, I gathered data from a large number of sources to fit a linear model.  The exogenous variables bear out a detailed explanation since all were statistically significant.

#### Demographic Factors
Two education factors were used <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a> the percentage of adults without a highschool diploma and the percentage of those with a bachelor's degree or higher.  In addition the county unemployment rate <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a>  was also used.  These economic indicators were combined with 2 religion factors.  From <a href=http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp>[8]</a> nearly 15,000 columns on the rate of denomination participation within county populations I used non-negative matrix factorization to reduce these columns to two latent topics.

<div>
    <a href="https://plot.ly/~SGShuman/49/" target="_blank" title="Religion NMF Feature vs. Target" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/49.png" alt="Religion NMF Feature vs. Target" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:49"  src="https://plot.ly/embed.js" async></script>
</div>


In fact, this NMF feature has good separation between red and blue counties as shown by the above figure.

#### State Level Effects
To counter state level effects, I used the 2012 state level partisan turnout <a href=http://bipartisanpolicy.org/library/2012-voter-turnout/>[9]</a> as a feature as well.  This provides a state level fixed effect, sort of like an intercept per state.

#### County Factors
The Rural-urban Continuum Code <a href=http://www.ers.usda.gov/data-products/county-level-data-sets.aspx>[7]</a> offers insight on the nature of population density inside a county.  Low scores indicate high population density somewhere inside the county and high scores indicate lower population density everywhere inside the county.

#### Historical Factors
The Cook score <a href=https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index>[10]</a>, which here I calucated on a county basis although it is typically on a state or congressional district basis, provides detail on the historical voting bias of a particular county.  The change in two party vote share from 2012 to 2008 [<a href=https://github.com/huffpostdata/election-2012-results>11</a>, <a href=https://www.udel.edu/johnmack/frec682/red_state_blue_state/>12</a>] also gives the model the ability to predict the trend from year to year.

<div>
    <a href="https://plot.ly/~SGShuman/59/" target="_blank" title="Historical Democratic Bias vs. Target" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/59.png" alt="Historical Democratic Bias vs. Target" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:59"  src="https://plot.ly/embed.js" async></script>
</div>


Interestingly the cook score is not entirely predictive of the results (see the big block of historically Republican counties in the blue), but does provide an interesting look at which party has more counties on the extreme and overall.

#### Campaign Expenditures and Donations
Campaign expenditures <a href=http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012>[13]</a> in a county summed over all monthly finance reports available offers some predictive power but donations <a href=https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/>[14]</a> weren't statistically significant in the result and so are not included in the model.

#### Office Data and Interaction
I obtained the locations of each of the 786 Obama offices and 284 Romney offices <a href=http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/>[2]</a> from John Sides who generously sent me a well formatted file.  Obama placed anywhere from 1 to 21 offices in a county (with by far most having one office), so I used the presence of 1 office or 2+ offices as dummy variables as well as interacting the Cook score with a boolean variable indicating the presence of an office.  Thus offices have three layers of importance: their presence, their quantity, and their interaction with the Cook score.  In the end, counties with larger Cook scores (indicating more partisan historical voting) had a larger effect from office placement.

I won't include a map here, but please refer to <a href='http://blogs.lse.ac.uk/usappblog/2014/05/12/when-placed-strategically-campaign-field-offices-can-be-very-important-in-turning-battleground-states-during-presidential-elections/'> Joshua Darr's and Matthew Levendusky's maps </a> for additional clarity <a href=http://apr.sagepub.com/content/42/3/529>[3]</a> on office placement.

### Linear Model Results
As found in previous papers, Democratic and Republican field offices offer starkly different in results.  While a field office placed by Democrats offers the chance for an increase in votes, Republican field offices have no statistical effect.  Obama's offices in my model were responsible for about 2.0M votes.

Altough these numbers are significantly higher than other researchers found, the effect of the interaction term in the model changes the magnitude of the effect, but not its importance.

### Simulation and Strategy
With the results of the linear model able to offer a statistical range of effects that are possible by placing a field office, I simulated the 2012 election 1000 times to see if any states could be flipped by placing field offices.  In this simulation, I assumed the Republican's field office bump was statistically significant.  I make this assumption because in 2016 it is possible that lessons learned from the 2012 campaign could cause equal effect to democratic strategy.

First the simulation calculates the number of votes per county without any field office placement.  Then the simulation places a limited number of field offices in counties, weighted by how close an election it was in the state the county is located.  Once the offices are placed, the vote increase is drawn from a normal distribution based off the coefficients and standard errors from the linear model.

<div>
    <a href="https://plot.ly/~SGShuman/53/" target="_blank" title="Average Percent Vote Increase by State" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/53.png" alt="Average Percent Vote Increase by State" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:53"  src="https://plot.ly/embed.js" async></script>
</div>

This leads to an average expected number of votes to be gained by state by adding field offices to every county.  We need to identify states where the projected difference in votes is covered by the difference between the right hand and left hand sides of the above graph.  In fact, since we can expect some statistical variation, we can identify all the counties who will flip in any simulation, even extreme ones.

This takes the counties that we would consider battlegrounds from a group looking like the below.

<div>
    <a href="https://plot.ly/~SGShuman/55/" target="_blank" title="Swing States - by Close Votes" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/55.png" alt="Swing States - by Close Votes" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:55"  src="https://plot.ly/embed.js" async></script>
</div>

And limits it to only these counties:
<div>
    <a href="https://plot.ly/~SGShuman/57/" target="_blank" title="Swing States - Simulation Results" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/57.png" alt="Swing States - Simulation Results" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:57"  src="https://plot.ly/embed.js" async></script>
</div>

As you can see (click to enable hover over), the expected swing states of North Carolina, Florida and Ohio all remain with in statistical contention, as well as New Hampshire.  This jives with some conventional wisdom about swing states.

#### Simulation Results
My model predicts every state correctly (Florida by only 10,000 votes)and indeed shows Obama flipping Florida with his effective field organization.  The other insight my model provides into Florida is that Democrats only win Florida in one out of every one thousand simulations when the Republicans mount a strong field organization there.  Obama did indeed flip it in 2012 but the Republicans could easily have answered with an effective field game of their own.

The big surprise is New Hampshire, where small changes can have a larger effect.  For every thousand times the simulation is run, New Hampshire will swith to the Republicans 250 times, while the other swing states will only swith 1 out of every 1000 times (hardly enough to focus a strategy there, at least on human timescales).

So, what is going on?  Why is New Hampshire, with all 4 of its electoral votes the key battleground for political strategy.

The answer is that in order to flip a state, one county must contain an appreciable percentage of the total state voting population and there must be relatively few counties so that the effect is not negated by random fluctuations in other counties.  When this is the case in New Hampshire some counties have a randomly small effect (possible with only 4 counties) and Hillborough County has a randomly strong effect, enough to turn the whole state.  In Callifornia, this isn't possible despite Los Angeles being so important in terms of population.

The visualization below shows this.  The size of the bubble is the size of the effect unscaled to population while the height is the percentage of the population that lives in that county.

<div>
    <a href="https://plot.ly/~SGShuman/61/" target="_blank" title="NH has one very influential county" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/61.png" alt="NH has one very influential county" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:61"  src="https://plot.ly/embed.js" async></script>
</div>


#### The Ideal Strategy
This simulation shows that field office placement could play a vital role in Republican strategy in 2012, with a more effective strategy getting out the vote in Florida, its 29 electoral votes can be turned.  With a longshot in NH (especially against Bernie Sanders), field offices can optimally change the single party turnout in a state by up to 5-7% but, in order for that to make a difference first the other party must not play, and secondly the margin must be sufficiently small that a state can be flipped.

In many of the swing states, if each party mounts an effective ground game, then nothing is gained or lost.  In Ohio, where there was a 3% in vote share difference in 2012 in favor of the Dems, has a 3% shift in favor of Obama given two effective ground games.  In states where the margin is less close, the ground game matters even less.

But much can change between now and election day. Some states may have smaller projected voting gaps (although wider is more likely). For Florida and New Hampshire and those hypothetical states this model can be used to find the most important counties to place field offices in. And help develop an overall strategy.


#### References
[1] The Ground Game in 2012, Seth Masket. Mischiefs of Faction  
http://www.mischiefsoffaction.com/2013/01/the-ground-game-in-2012.html

[2] How Much Did the 2012 Air War and Ground Game Matter?, John Sides. The Monkey Cage  
http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/

[3] Relying on the Ground Game: The Placement and Effect of Campaign Field Offices, Joshua P. Darr, Matthew S. Levendusky. American Politics Research May 2014 vol. 42 no. 3 529-548  
http://apr.sagepub.com/content/42/3/529

[4] Did Obama's Ground Game Matter?, Seth E. Masket. Public Opin Q (2009) 73 (5): 1023-1039.  
http://poq.oxfordjournals.org/content/73/5/1023.full

[5] Campaign Field Offices and Voter Mobilization in 2012, Aaron C. Weinschenk. Presidential Studies Quarterly 45, no. 3 (September)  
http://www.acweinschenk.net/uploads/1/7/3/6/17361647/psq12213.pdf

[6] Office Space: A Geo-Spatial Analysis of the Effects of Field Offices on Voter Turnout, Seth Masket et al. American Political Science Association, August 28-31, 2014, Washington, D.C.  
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

[14] Political influence by county: A new way to look at campaign finance data, Ryan Sibley, Bob Lannon, Ben Chartoff. Sunlight Foundation Blog  
https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/