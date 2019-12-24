<h1>Film Finder</h1>
<h3>The movie recommendation tool based on similar content.</h3>

The Film Finder accepts any number of movie titles as imputs, and outputs a user-selected amount of movies similar to those inputted.

For example, the movie Jaws and the movie Black Water are both similar because they are both about danger in the water.  Thus, the movie plots for each movie on Wikipedia will contain a lot of words related to this idea.  The algorithm for this recommendation tool uses the text from the movie's Wikipedia page "Plot" section to compares moves to each other.

I downloaded the dataset from Kaggle (https://www.kaggle.com/jrobischon/wikipedia-movie-plots) and used Python to create the system.