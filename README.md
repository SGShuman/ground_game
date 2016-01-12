# Creating the Ideal Field Office Strategy for the Presidential Election
#### An analysis of the 2012 presidential election with an eye towards 2016

### Introduction
The folk wisdom of the 2008 and 2012 elections is that the democrat's superior 'ground game' handed them the crucial swing states needed for an electoral victory.  By mobilizing 786 field offices [1] located in key counties, grass roots supporters were able to get out the vote and just edge Obama over the top.  But how much truth is there to this traditional narrative?

In a previous analyses by John Sides and Lynn Vavrek [2], field offices were found to have had a strong role in winning Florida for the democrats and along with other factors gaining about 248,000 votes nationally.  Similarly Joshua Darr, Luke Keele, and Matthew Levendusky [3] have estimated that Obama's 2008 field operation won him about 275,000 votes and Seth Masket [4] found ground game influential in North Carolina, Florida, and Indiana.

These analyses have typically relied on factors like median income, unemployment, demographic factors and historical voting patterns [3, 5] to regress the share of the two party vote each candidate receives.  However, recent analysis [6] has shown that office placement is  important for one party turnout percentage.

Given that a field office can increase turnout, I modeled single party turnout percentage based on a set of similar factors to try to identify which counties are especially susceptible to the turnout effect.  I find that counties with a historical voting bias are more susceptible to a large turnout effect for the historically disenfranchised party and present a strategy of office placement to maximize this effect.

### Methodology
In order to understand the effect of a field office being placed in a county, I gathered data from a large number of sources to fit a linear model.  The exogenous variables bear out a detailed explanation since all were statistically significant.

#### Demographic Factors
Two education factors were used [7] the percentage of adults without a highschool diploma and the percentage of those with a bachelor's degree or higher.  In addition the county unemployment rate [7] was also used.  These economic indicators were combined with 2 religion factors.  From [8] nearly 15,000 columns on the rate of denominations within county populations I used non-negative matrix factorization to reduce these columns to two latent topics.

<div>
    <a href="https://plot.ly/~SGShuman/51/" target="_blank" title="Religion NMF Feature vs. Target" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/51.png" alt="Religion NMF Feature vs. Target" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:51"  src="https://plot.ly/embed.js" async></script>
</div>


In fact, this NMF feature has good separation power as shown by the above figure.

#### State Level Effects
To counter state level effects, I used the 2012 state level partisan turnout [9] as a feature as well.  This provides a state level fixed effect, sort of like an intercept per state.

#### County Factors
The Rural-urban Continuum Code [7] offers insight on the nature of population density inside a county.  Low scores indicate high population density somewhere inside the county and high scores indicate lower population density everywhere inside the county.

#### Historical Factors
The cook score [10], which here I calucated on a county basis although it is typically on a state or congressional district basis, provides detail on the historical voting bias of a particular county.  The change in two party vote share from 2012 to 2008 [11, 12] also gives the model the ability to predict the trend from year to year.

<div>
    <a href="https://plot.ly/~SGShuman/51/" target="_blank" title="Historical Democratic Bias vs. Target" style="display: block; text-align: center;"><img src="https://plot.ly/~SGShuman/51.png" alt="Historical Democratic Bias vs. Target" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="SGShuman:51"  src="https://plot.ly/embed.js" async></script>
</div>

Interestingly the cook score is not entirely predictive of the results (see the big block of historically republican counties in the blue), but does provide an interesting look at which party has more counties on the extreme.

#### Campaign Expenditures and Donations
Campaign expenditures [13] in a county summed over all monthly finance reports available offers some predictive power but donations [14] weren't statistically significant in the result and so are not included in the model.

#### Office Data and Interaction
I obtained the locations of each of the 786 Obama offices and 284 Romney offices [2] from John Sides who generously sent me a well formatted file.  Obama placed anywhere from 1 to 21 offices in a county (with by far most having one office), so I used the presence of 1 office or 2+ offices as dummy variables as well as interacting the cook score with a boolean variable indicating the presence of the office.  Thus offices have three layers of importance: their presence, their number, and their interaction with the cook score.  In the end, counties with larger cook scores (indicating more partisan historical voting) had a larger effect from office placement.

I won't include a map here, but please refer to <a href='http://blogs.lse.ac.uk/usappblog/2014/05/12/when-placed-strategically-campaign-field-offices-can-be-very-important-in-turning-battleground-states-during-presidential-elections/'> Joshua Darr's and Matthew Levendusky's maps </a> for additional clarity [15].

### Linear Model Results
As found in previous papers, democratic and republican field offices offer stark different in results.  While a field office placed by democrats offers the chance for an increase in votes, republican field offices have no statistical effect.  Obama's offices in my model were responsible for about 2.0M votes and Romney's offices were responsible for 0.5M votes.  The effect of the republican offices was not statistically significant though.  Altough these numbers are significantly higher than other researchers found, the effect of the interaction term in the model greatly changes the magnitude of the effect, but not its importance.

### Simulation and Strategy
With the results of a linear model able to offer a range of the effects that are possible by placing a field office, I simulated the 2012 election 1000 times to see if any states could be flipped by placing field offices.  In this simulation, I assumed the republican's field office bump was statistically significant.  I make this assumption because in 2016 it is possible that lessons learned from the 2012 campaign could cause equal effect to democratic strategy.

First the simulation calculates the number of votes per county without any field office placement.  Then the simulation places a limited number of field offices in counties, weighted by how close an election it was in the state the county is located.  Once the offices are placed, the vote increase is drawn from a normal distribution based off the coefficients and standard errors from the linear model.

In the simulation



#### References
[1] http://www.mischiefsoffaction.com/2013/01/the-ground-game-in-2012.html

[2] http://themonkeycage.org/2013/05/how-much-did-the-2012-air-war-and-ground-game-matter/

[3] http://www.personal.psu.edu/ljk20/GroundGame.pdf

[4] http://poq.oxfordjournals.org/content/73/5/1023.full

[5] http://www.acweinschenk.net/uploads/1/7/3/6/17361647/psq12213.pdf

[6] http://static1.1.sqspcdn.com/static/f/864938/25350207/1408716516677/Allen+et+al.+APSA+2014.pdf?token=8SQWbYH63TzRnaVXi9JXF1mWlTY%3D

[7] http://www.ers.usda.gov/data-products/county-level-data-sets.aspx

[8] http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp

[9] http://bipartisanpolicy.org/library/2012-voter-turnout/

[10] http://cookpolitical.com/house/pvi

[11] https://github.com/huffpostdata/election-2012-results

[12] https://www.udel.edu/johnmack/frec682/red_state_blue_state/

[13] http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2011_2012

[14] https://sunlightfoundation.com/blog/2013/10/23/political-influence-by-county-a-new-way-to-look-at-campaign-finance-data/

[15] http://blogs.lse.ac.uk/usappblog/2014/05/12/when-placed-strategically-campaign-field-offices-can-be-very-important-in-turning-battleground-states-during-presidential-elections/