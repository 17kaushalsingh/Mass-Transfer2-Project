# Mass Transfer 2 Project

## Problem Statement
Experimental data collection remains expensive and time consuming.
Hence it is important to create a digital twin of many process systems in industries.
These digital twins can recreate synthetic data that can be useful for making surrogate models using neural networks and other methods
Create software (preferably with a graphic user interface, use Python and Python-frameworks for coding and programming.
Solve the liquid-liquid extraction / solid liquid extraction problem through the solution of a system of equation, so that a large number of stages can be handled. Generalize the problem solving as much as possible.

## Software Requirements
The software is expected to have the following components that are necessary for parameter optimization.

1. A simulation framework for solving multistage crosscurrent and countercurrent extraction (either liquid-liquid or
solid liquid system) and comparison of the two processes.

fsolve in python: A simulation framework for solving crosscurrent and countercurrent solid liquid
extraction using numerical method/ solvers [example: fsolve function in MATLAB or python] for solution of a
system of non-linear algebraic equation. [ More generalized framework for large number of stages]. Initially to test
the solutions, you can use the guess values close to the solutions obtained from graphical solutions. Also plot the
equilibrium data to fit the polynomial through the LL equilibrium curve in the right angles triangle method. [No
marks will be given if the graphical methods are used]

A graphical user interface can be useful so that the user can choose the option of entering various equilibrium data
and can opt for crosscurrent or countercurrent operation.

2. Make a heat map showing the composition in raffinate/ extract or overflow/underflow composition . Also show the
heatmap of the flowrate of r raffinate and extract composition] and flow rate for each stage for any number of
stages. It should also show the percentage removal at each of the stages.

3. Generation of response surface and optimization of the process. Create data in the form of input output to construct
a surrogate model. [ for example: input 1= number of stage, input 2=solvent amount and input 3= initial feed
composition and output 1=percentage removal of solute].

4. Fit a neural network model (choose the number of nodes and number of layers, ANN or RNN) that can predict the
percentage removal or sequential percentage removal in multi-stage operation. Create a response surface of
percentage removal as a function of two inputs at a time to create a 3D surface and contour plot. Such plots are
necessary for optimization of process parameters.
Use the following data for various groups to use the equilibrium data and solution. However, you need to make a
software that can be used to calculate the percentage removal for any number of stages.
Write the contribution of each member clearly. Each of the group member should have a distinct code that needs
to be presented in the PPT. They need to be integrated into making a common software.
Credit will be given if your algorithm is having some novelty:
Presentation:
Flowchart summary of the work [20].

For each group member, explanation of the code and explanation of the variables used for the model [100]. Every
variable, loops to be explained and function used to be explained.
Plot of results from the software (the output in heatmap form , and explanation) for each group member (goodness
of the fit from regression), ( Prediction and actual data), (Surace plot and optimal point) [ 80].

DATA:
Group 9 : Cottonseed oil system: (A) Liquid Propane (B) Oleic acid (C) at 98.5 degree C, 625 lb/in2 abs. 
Smoothed equilibrium tie-line data, in weight percent as follows in (data.json)

i) Plot the equilibrium data on the following coordinate system: (a) N against X and Y; (b) X against Y.

ii) If 100 kg of a cottonseed oil-oleic acid solution containing 25% acid is to be extracted twice in crosscurrent fashion, each
time with 1000 kg of propane, determine the compositions, percent by weight, and the weights of the mixed extracts and
the final raffinate. Determine the compositions and weights of solvent-free products. Make the computations on the
coordinates plotted in the previous part (a).

iii) If 1000 kg/h of a cottonseed oil-oleic acid solution containing 25% acid is to be continuously separated into products
containing 2 and 90% acid (solvent-free compositions) by countercurrent extraction with propane, make following
computations on the coordinate system of parts (a) and (b):
(a) What is the minimum number of theoretical stages required?
(b) What is the minimum external extract-reflux ratio required?
(c) For an external extract-reflux ratio of 4.5, determine the number of theoretical stages, the position of the feed stage,
and the quantities, in kg/h, of the following streams: E1, BE, E’, R0, RNp, PE’and S
(d) What do the equilibrium data indicate as to the maximum purity of oleic acid that could be obtained?