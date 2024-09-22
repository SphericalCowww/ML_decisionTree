# Decision Tree Supervised Classification on 2D Point
The code is using the decision tree method to classify 2D tagged data into 2 categories. The helper functions in the code should work for both higher feature and category dimensions, just that those would be difficult to illustrate on a 2D plot. The categories however, should hold discrete values, like blue/red in the following example.

The example data is given as the following, where each data point is tagged as a blue cross or a red circle:

<img src="https://github.com/SphericalCowww/ML_decisionTree/blob/main/input2DPlot0Data_Display.png" width="320" height="240">

The decision tree code runs on python3:

    python3 decisionTree.py

<img src="https://github.com/SphericalCowww/ML_decisionTree/blob/main/input2DPlot1DecTree_Display.png" width="320" height="240">

The code outputs the image above, where the 2D region is partitioned into blue and red regions. Future points that fall into the blue (red) region will be categorized as a cross (circle).

As for the gradient-boosted decision tree code:

    python3 gradientBoostedDecisionTree.py

<img src="https://github.com/SphericalCowww/ML_decisionTree/blob/main/input2DPlot2Boosted_Display.png" width="320" height="240">

The code outputs the image above, where the 2D region now has gradient blue and red regions. The gradient toward blue (red) means that future points that fall in this region are preferably classified as cross (circle).

The more advanced variations, such as random forest (bootstraps feature dimensions, so need more than 2 features as in the example to be effective; good at handling missing data) and AdaBoost (adaptive gradient boost, which is an upgrade to the random forest by introducing weights on the trees in a reasonable way), have not yet been implemented.

## Key Words:
- recursion, greedy algorithm, dynamic programming, <a href="https://stackoverflow.com/questions/6184869/what-is-the-difference-between-memoization-and-dynamic-programming">memoization</a>/<a href="https://www.geeksforgeeks.org/tabulation-vs-memoization/">tabulation</a>
## References:
- StatQuest with Josh Starmer's Youtube channel (<a href="https://www.youtube.com/watch?v=_L39rN6gz7Y">Youtube1</a>, <a href="https://www.youtube.com/watch?v=g9c66TUylZ4">2</a>, <a href="https://www.youtube.com/watch?v=3CC4N4z3GJc">3</a>, <a href="https://www.youtube.com/watch?v=J4Wdy0Wc_xQ">randForest</a>, <a href="https://www.youtube.com/watch?v=LsK-xG1cLYA">adaBoost</a>)
