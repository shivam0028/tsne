# t-Distributed Stochastic Neighbor Embedding (t-SNE)

Implementation of the paper [**Visualizing data using t-SNE**](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbclid=IwAR0Bgg1eA5TFmqOZeCQXsIoL6PKrVXUFaskUKtg6yBhVXAFFvZA6yQiYx-M)

t-SNE visualizes high-dimensional data by giving each datapoint a location in 2-D or 3-D map. It is a variant of Stochastic Neighbor Embedding (SNE) that produces better visualization by reducing the tendency to crowd points together in the center of the map. The aim of dimensionality reduction is to preserve as much of the significant structure of high-dimensional data as possible in the low-dimensional map. This algorithm called the ”t-SNE” is capable of capturing the local structure of the high-dimensional data very well, while also revealing the global structure such as clusters.

t-SNE works by minimizing the mismatch between the datapoints relationship defined using probability distribution, which is defined using gaussian distribution in high-dimensional space and Student-t distribution in the low-dimensional space.

## Installing the requirements
```
pip install -r requirements.txt
```

## Reference
<a href="https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbclid=IwAR0Bgg1eA5TFmqOZeCQXsIoL6PKrVXUFaskUKtg6yBhVXAFFvZA6yQiYx-M">[1]</a>
Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.11 (2008).
