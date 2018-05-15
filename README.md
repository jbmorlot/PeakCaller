# ZINB Peak Caller
Peak caller based on the Zero Inflated Negative Binomial (ZINB) distribution [Cusco &al 2016 Bioinformatics](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btw336).

The algorithm works in two phases:
 1. Fit the distribution parameters on the entire vector by using multiple random initialization.
 2. Fit the distribution parameters on widows of size WS using the optimization on the entire vector
as initialization.

## How to use

To get a binary vector with a threshold at a p-value of <img src="https://latex.codecogs.com/gif.latex?10^{-5}" title="10^{-5}" /></a>
```
vectB = Peaks_Caller(vect)
```

To get the p-value vector
```
vectB = Peaks_Caller(vect,getPval=True)
```

Note:By default, the windows are compute in parallel, using all the cores of the machine.
To change it, change Ncore parameter.
