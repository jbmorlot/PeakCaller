# Peak_Caller
Peak caller based on the Zero Inflated Negative Binomial (ZINB) distribution (Cusco &al 2016 Bioinformatics).
The algorithm works in two phases:
First fit the distribution parameters on the entire vector by using multiple random initialization.
Then fit the distribution parameters on widows of size WS using the optimization on the entire vector
as initialization.
##How to use
To get a binary vector with a threshold at a p-value of $10^{-5}$
```
vectB = Peaks_Caller(vect)
```
To get the p-value vector
```
vectB = Peaks_Caller(vect,getPval=True)
```

Note:By default, the windows are compute in parallel, using all the cores of the machine.
To change it, change Ncore parameter.
