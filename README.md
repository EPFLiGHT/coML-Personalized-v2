# Semester project on private and personalized ML
semester project felix grimberg - private robust and personalized ML

Read about the aim and objectives of this project in the [report](Report-200616b.pdf) (also on [overleaf](https://www.overleaf.com/read/qpymtfymrfzy)).

In this repo, we investigate methods for automatic data selection based on the distance between gradients in SGD-based optimization algorithms.

## To view correctly displayed formulas directly in GitHub, open [the IPYNB version of this README](README.ipynb)!
It's updated nightly to mirror the contents of the Markdown file.

### Using the repo (no longer actually using pipenv)

We use `pipenv` to control external package dependencies. The safest way to avoid dependency issues while running code from this project is to follow these steps:

1. Make sure the latest version of `pipenv` is installed:
  1. ```pip install --upgrade pipenv```
  2. If necessary, add the `--user` option. When using this option, **remember to update the `PATH` environment variable** with the directory in which pipenv was installed:
  
    ```export PATH=<directory>:$PATH```
2. Clone the repository (`git clone`) and `cd` into it.
3. Install the dependencies from the `Pipfile.lock` file: ```pipenv sync```
4. Make sure to use the virtual environment you've just created:
  1. Activate the virtual environment: `pipenv shell`
  2. To execute IPython Notebooks, you need to create a Kernel for this environment. With the virtual environment active, execute the following (you can pick the venv name):
  
  ```python -m ipykernel install --user --display-name <venv_name> --name <venv_name>```
5. Obtain the data, which I do not own, and place it in `../data/private/`

### Files:

`data-preprocessing.ipynb`: This notebook handles the pre-processing of the Ebola dataset. Obtain the dataset `csv` file from Mary-Anne Hartley and place it in a folder named `../data/private` (relative to the repo folder). Then, execute all cells of `data-preprocessing.ipynb` in order to deal with missing values. The pre-processing is largely copied from Annie's 2017 paper on *Predicting Ebola Infection*.

`global-model-1.ipynb`: This notebook creates the feature matrix $\mathbf{X}$ and the target vector $\mathbf{y}$ for the EVD(+)/EVD(-) prediction task from Annie's 2017 paper. Execute the cells to create the files used as input for the federated learning simulation. In the second part of the notebook, we train a predictive model on all data using `scikit-learn`.

`Titanic-preprocessing.ipynb`: This notebook handles the loading and pre-processing of the Titanic dataset. Install `tensorflow_datasets` and execute all cells of `Titanic-preprocessing.ipynb` to produce the clean data files used in the federated learning simulations. Many ideas in this notebook are inspired from https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python.

`Jax_federated_learning.ipynb`: This notebook is the heart of the project. It implements the Weight Erosion aggregator in an earlier version of Sai Praneeth Karimireddy's Jax-based FL simulation framework (GitHub: @saipraneet). Jump to the Markdown cell titled *Run a federated learning simulation* to specify all simulation parameters, then execute all cells in order (from the top) to run the simulation and output its results.

`modules`: This folder contains two `Python` files with functions and classes used in `Jax_federated_learning.ipynb`.

### Ways ahead:

- Automate hyper-parameter tuning and use cross-validation
- Split the data set in more different ways
  - IID
  - Vary number of clients
  - By gender
  - By outcome (label skew)
  - skew $\mathbb{P} \left( \mathbf{x} | y \right)$
  - skew $\mathbb{P} \left( y | \mathbf{x} \right)$
- Test the gradient-based method on a different task (maybe more parameters and much larger data set?)
- Implement other methods
- Refine gradient-based similarity some more, try it out on other problems
- Rigorously define similarity (e.g., with learning theory notion of discrepancy)

### Setting:

We consider a network of participants $u_i$, each collecting samples from an underlying distribution $\mathcal{D}_i$.
One participant $u_0$ is called the user. This participant wishes to perform an inference task, such as (regularized) linear or logistic regression, to gain knowledge about the distribution $\mathcal{D}_0$ from which they collect data.
The user $u_0$ further believes that *some, but not all,* of the other participants $u_i$ collect sufficiently interoperable data from sufficiently similar underlying distributions, that the samples $\mathcal{S}_i$ collected by them could help reduce the *true loss* of $u_0$'s model on $\mathcal{D}_0$, if the samples $\mathcal{S}_i$ are included in the training of the regression model.
Our task is to define an algorithm which can help the user $u_0$ minimize the expected *true error* of the fitted regression model on $\mathcal{D}_0$ by selecting a subset of participants whose collected samples are used during training.

As an example, the participants could be individual hospitals dispersed across one or several regions. Doctors could have reasons to expect that the same set of symptoms is more strongly associated with one diagnosis for the patients of one hospital, and with another diagnosis for the patients of a different hospital.
As a layman example, it seems conceivable that diseases caused by poor sanitation are more likely to occur in poor rural regions, while drug abuse is more frequent in more affluent urban settings - while causing similar symptoms. This is not to say that the symptoms are indeed similar.
Knowing this, a medical professional from one hospital might want to train a diagnostic model, recognizing that they would benefit from enhancing the limited number of samples collected at their own hospital with samples collected at a selected subset of other hospitals.

### Why we investigate methods based on the distance between gradients in SGD-based optimization algorithms:

We say that a distribution $\mathcal{D}_i$ is *similar to $\mathcal{D}_0$ with respect to the specific inference task at hand* (in short: *similar*), if the weights of each distribution's true model for this inference task are similar.

- [ ] We might want to define a metric for that notion of *similarity* $\uparrow$

Recall that, given an inference task defined by:
- a class of models $\mathbb{M}$ and
- a loss function $\mathcal{L} (y, \hat{y})$,

the true model $\mathcal{M}_{i}^{true}$  minimizes the expected loss $\mathcal{R}_{\mathcal{D}_i} \left( \mathcal{M} \right)$ on the distribution $\mathcal{D}_i$:

$ \mathcal{R}_{\mathcal{D}_i} \left( \mathcal{M}_{i}^{true} \right) \triangleq \underset{(\mathbf{x}, y) \in \mathcal{D}_i}{E} \left[ \mathcal{L} \left( y, \mathcal{M}_{i}^{true} \left( \mathbf{x} \right) \right) \right] \leq 
\mathcal{R}_{\mathcal{D}_i} \left( \mathcal{M}_{i}^{other} \right)
\forall \mathcal{M}_{i}^{other} \in \mathbb{M} $

Let $\mathcal{M}^{tentative} \in \mathbb{M}$ be a tentative model which performs very suboptimally on two **similar** distributions $\mathcal{D}_i$ and $\mathcal{D}_j$. In particular let:

$ \textbf{min} \left( \mathcal{R}_{\mathcal{D}_i} \left( \mathcal{M}^{tentative} \right),  \mathcal{R}_{\mathcal{D}_j} \left( \mathcal{M}^{tentative} \right) \right) \gg \textbf{max} \left( \mathcal{R}_{\mathcal{D}_i} \left( \mathcal{M}^{true}_j \right),  \mathcal{R}_{\mathcal{D}_j} \left( \mathcal{M}^{true}_i \right) \right)$

Further, let $\mathbf{g}_i$ and $\mathbf{g}_0$ be the gradients of the loss function $\mathcal{L} \left( y_i, \mathcal{M}^{tentative} \left( \mathbf{x}_i \right) \right)$, respectively $\mathcal{L} \left( y_0, \mathcal{M}^{tentative} \left( \mathbf{x}_0 \right) \right)$, for a sample $(y_i, \mathbf{x}_i \in \mathcal{D}_i)$, respectively $(y_0, \mathbf{x}_0 \in \mathcal{D}_0)$, with respect to the model weights.
Then it follows that, in expectation over all possible samples from each distribution, $\mathbf{g}_i$ and $\mathbf{g}_0$ will be similar as well.

- [ ] We might want to prove that $\uparrow$, for which we would also need to be more rigorous about what we mean by the two gradients being similar.
- [ ] We haven't shown that the gradients of two *more similar* distributions will be more similar on average than the gradients of two *less similar* distributions.
- [ ] Here we assume that $\mathcal{M}^{tentative}$ performs very suboptimally on both distributions. What happens as $\mathcal{M}^{tentative}$ approaches, let's say, the *true model of the joint distribution*? Clearly, the gradients will begin to diverge. Can we formalize this in some way? In fact, I believe that the true model of the joint distribution is such that the sum $\mathbf{g}_i + \mathbf{g}_0$ adds up to $\mathbf{0}$ in expectation.
- [ ] Related to the previous point: Maybe the aggregated similarity estimator (cf. below) should experience some form of decay, if we expect the (expected) gradients to diverge as the model gets better? But then, if we do enough SGD steps, all similarities would end up decaying at some point and we would eventually end up with the local model again. So where do we find the compromise here?
- [ ] Is there a canonical symbol to denote the *true error* of a model? That's the correct name for what I'm defining here, right?
- [ ] I'm almost certain that *true model* is not the correct term for the model that minimizes the true error over all models in a class of models. What's the correct term?
- [ ] We might want to distinguish between the norm and the angle of the gradients. Though a gradient that points very far or not-far-at-all in the right direction still is not useful and certainly does not hint at *similar underlying distributions.*

It is therefore our intention to invent an estimator for the (so far vaguely defined) similarity of two distributions based on the distance between gradients in SGD-based optimization algorithms involving samples taken from these two distributions.
It is clear that this estimator must be aggregated over a number of SGD or minibatch steps due to the random nature of the SGD / minibatch gradient.

### Methods:

This whole thing about similar distributions leading to similar gradients holds *in expectation*, but the individual gradients can randomly vary a lot if the minibatch size is small.
It is thus clear that we must come up with a similarity estimator that's aggregated over time.

The following could be a rough sketch of the algorithm:
1. Perform a certain number of steps with all participants until we've aggregated a stable enough similarity estimator.
2. Select a subset of the available data sets, potentially weighing the contribution of each data set according to the estimated similarity of its underlying distribution.
3. Train the model on the selected data sets until some stopping criterion (convergence, number of communication rounds, ...) is reached. Step 2 could potentially be repeated at regular intervals throughout step 3.

There is a balance to be struck between the number of communication rounds needed to perform steps 1 & 3 satisfactorily, and the requirement of *communication efficiency*.

I don't know whether weight quantization and model sparsification are viable strategies to achieve communication efficiency in the training of linear models.
