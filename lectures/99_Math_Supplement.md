# Math Supplement

## Sample Statistics

### Sample Mean

$$\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$$

$x_1$ is the value of observation $i$

$N$ is the total number of observations

Also commonly denoted as $m$ and rarely as $\hat{\mu}$ (the estimate of the population mean). The sample mean is a measure of centrality, and an _unbiased estimator_ of the population mean $\mu$ - that is, $\lim_{N\rightarrow \infty} \bar{x} = \mu$. However, unlike the sample median, the sample mean is not robust to outliers in the data.

### Sample Variance

$$s^2 = \frac{\sum(x_i-\bar{x})^2}{N-1}$$

$\bar{x}$ is the sample mean

$x_1$ is the value of observation $i$

$N$ is the total number of observations

Note that the $N-1$ term is called Bessel's correction, and turns the sample variance from a biased estimator (which would just use a denominator of $N$) to an unbiased one. One way to think about about it is that the denominator represents the degrees of freedom of the parameter in question ($s^2$). Normally, this would just be equal to the number of observations ($N$), but because we use the sample mean $\bar{x}$ in our calculation, we have already removed one degree of freedom.

### Sample Standard Deviation

$$s = \sqrt{\frac{\sum(x_i-\bar{x})^2}{N-1}}$$

### Sample Median

For a set of values ordered values $x_1 \leq x_2 ... \leq x_N$, if $N$ is odd, the median is $x_{(N+1)/2}$, and if $N$ is even, the median is $\frac{x_{N/2}+x_{N/2+1}}{2}$. There does not appear to be a single, agreed-upon variable for median, and it is variously written as $m$ (to be confused with mean), $med$, or simply $median$.

The sample median is a measure of centrality, but whether it is biased or unbiased as an estimator of the population median depends on the distribution of the population. It _is_ unbiased in the case of symmetric distributions like the normal distribution; it also happens to be an unbiased estimator of the population _mean_ in that case.

## Common Distributions

### Normal Distribution

The probability density function of the normal distribution is given by

$$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

$\sigma$ is the population standard deviation

$\mu$ is the population mean

It is also commonly written as $\mathscr{N}(\mu,\sigma)$. Lots of things in nature seem to follow this distribution, and much ink has been spilled on its properties.

You may also encounter the _Standard Normal_ distribution, which takes the form $\mathscr{N}(\mu=0,\sigma=1)$, and is the basis for the $z$ test.

### $t$ Distribution

The $t$ distribution (or Student's $t$ distribution) is actually a family of distributions, and has the probability density function

$$f(t) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1+\frac{t^2}{\nu})^{-(\nu+1)/2}$$

$\Gamma$ is the gamma function

$\nu$ is the degrees of freedom

In practice, you will never use this function directly, and are generally interested in one-tailed (integrated from $-\infty$ to $t$ or from $t$ to $+\infty$) or two-tailed (integrated from $-t$ to $+t$) cumulative distribution functions of this distribution, which give you the $p$ value (or $1-p$) of a $t$ test.

It is important, however, to note that the $t$ distribution is parameterized by the value of $\nu$, which is affected by experimental design through $N$, the number of samples collected. 

The $t$ distribution approaches the standard normal distribution as $\nu \rightarrow \infty$, meaning the $t$ test also approaches the $z$ test in the same way. The further $\nu$ is from $\infty$, the more probability mass is contained in the tails of the distribution. In practice, that means the fewer samples you have, the greater probability this distribution gives to extreme outcomes.

<!-- ### F Distribution

TODO -->

## Other Concepts

<!-- ### Central Limit Theorem

TODO -->

### Types of Data

Data generally come in four scales: nominal, ordinal, interval, and ratio. It is important to know what kind of data we are dealing with when thinking about what kind of analysis to perform. Also, a nice property of these scales is that they are, in fact, ordered in such a way so that you can always treat data from one scale as a type that preceded it.

___Nominal___ data are data that come in categories. The categories do not have a natural order or ranking, and the typical measures of centrality or dispersion do not apply. Blood types are an example of nominal data - there is no "mean" or "median" blood type. Mode is about as close as you can get to a measure of centrality, though really, it's a measure of frequency.

___Ordinal___ data are data that are on an ordered scale, where the differences between elements on the scale are not well-defined. For example, rankings, education levels, Likert scales, pain scales, etc are ordinal data. For example, knowing that racecar A came before racecar B tells you ordinal information, but doesn't tell you far apart they were in time. Medians and percentiles start to be meaningful from here on out.

___Interval___ data are data that are on an ordered scale where the differences between elements on the scale _are_ well-defined and equal, but where zero does not mean "none" of what you're measuring - that is, you cannot calculate ratios for that data. Examples include IQ scores, Fahrenheit, Celsius (but not Kelvin!)[^FCconfusion], time of day (but not duration!), and so on. Means and standard deviations start to be meaningful.

[^FCconfusion]: Wait, but there is definitely a zero point in F and C! Well, yes, but the choice of zero for these two was arbitrary in comparison to what the scales are supposed to measure (average heat energy - or temperature). 40 degrees is not twice as hot as 20 degrees (on either F or C). Kelvin _is_ tied directly to the idea of "zero heat" (absolute zero), so it is a ratio value. Same goes for time of day or calendar dates, even though we have zeros for those as well. Addition and subtraction start being meaningful here.

___Ratio___ data are the data that are interval, but _do_ have a well-defined zero value. Kelvin, time durations, weight, length, etc are ratio data. Ratios (obviously), and multiplication start being meaningful.

Although parametric statistics are built on interval and ratio assumptions, ordinal data this assumption can often be bent to allow for the inclusion of ordinal data when you have enough of it.
