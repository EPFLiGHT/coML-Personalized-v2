# semester-project-privateML
semester project felix grimberg - private robust and personalized ML

Find some information about the aim and objectives of this project in the report.

In this repo, we investigate methods for automatic data selection based on the distance between gradients in SGD-based optimization algorithms.

### Setting:

We consider a network of participants $u_i$, each collecting samples from an underlying distribution $\mathcal{D}_i$.
One participant $u^*$ is called the user. This participant wishes to perform an inference task, such as (regularized) linear or logistic regression, to gain knowledge about the distribution $\mathcal{D}^*$ from which they collect data.
The user $u^*$ further believes that _some, but not all,_ of the other participants $u_i$ collect sufficiently interoperable data from sufficiently similar underlying distributions, that the samples $\mathcal{S}_i$ collected by them could help reduce the _true loss_ of $u^*$'s model on $\mathcal{D}^*$, if the samples $\mathcal{S}_i$ are included in the training of the regression model.
Our task is to define an algorithm which can help the user $u^*$ minimize the expected _true error_ of the fitted regression model on $\mathcal{D}^*$ by selecting a subset of participants whose collected samples are used during training.

As an example, the participants could be individual hospitals dispersed across one or several regions. Doctors could have reasons to expect that the same set of symptoms is more strongly associated with one diagnosis for the patients of one hospital, and with another diagnosis for the patients of a different hospital.
As a layman example, it seems conceivable that diseases caused by poor sanitation are more likely to occur in poor rural regions, while drug abuse is more frequent in more affluent urban settings - while causing similar symptoms. This is not to say that the symptoms are indeed similar.
Knowing this, a medical professional from one hospital might want to train a diagnostic model, recognizing that they would benefit from enhancing the limited number of samples collected at their own hospital with samples collected at a selected subset of other hospitals.