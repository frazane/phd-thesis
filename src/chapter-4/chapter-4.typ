#import "@preview/lovelace:0.2.0": *
#show: setup-lovelace

= Sub-kilometer surface weather modelling <chap-4>

This chapter has been adapted from the manuscript:
\ \
#set list(indent: 20pt)
- Zanetta, F., D. Nerini, M. Buzzi, H. Moss 2024: Efficient modeling of sub-kilometer surface wind with Gaussian processes and neural networks, *under review*. 
\
It addresses the topic of spatiotemporal coherence and the blending of multiple sources of information for the modelling of surface weather variables. This manuscript presents a novel methodology that combines neural networks with Gaussian processes regression to optimally combine real-time observations with post-processed NWP data. Additionally, the proposed method leverages scalable covariance modelling techniques, allowing the generation of spatially coherent realizations of surface weather variables fields.

#pagebreak()

== Summary <4-summary>
Accurately representing surface weather at the sub-kilometer scale is crucial for optimal decision-making in a wide range of applications. This motivates the use of statistical techniques to provide accurate and calibrated probabilistic predictions at a lower cost compared to numerical simulations. Wind represents a particularly challenging variable to model due to its high spatial and temporal variability. This paper presents a novel approach that integrates Gaussian processes (GPs) and neural networks to model surface wind gusts, leveraging multiple data sources, including numerical weather prediction (NWP) models, digital elevation models (DEM), and in-situ measurements. Results demonstrate the added value of modeling the multivariate covariance structure of the variable of interest, as opposed to only applying a univariate probabilistic regression approach. Modeling the covariance enables the optimal integration of observed measurements from ground stations, which is shown to reduce the continuous ranked probability score compared to the baseline. Moreover, it allows the direct generation of realistic fields that are also marginally calibrated, aided by scalable techniques such as Random Fourier Features (RFF) and pathwise conditioning. We discuss the effect of different modeling choices, as well as different degrees of approximation, and present our results for a case study.

== Introduction <4-introduction>
Surface weather affects a wide range of human activities. Accurate modeling of surface weather phenomena is beneficial at all societal levels, from individuals to critical infrastructure. Among these phenomena, surface winds play a crucial role, as they can account for impactful weather conditions such as dangerous storms or even winds harnessable for energy production. Consequently, motivated by a desire for optimal decision-making under surface weather uncertainty, there is a strong demand for accurate and precise information on the likely distribution of surface winds. However, the phenomena of surface winds, as characterized by high spatial and temporal variability and the sparseness and limited representativeness of our measurement records, challenge our traditional modeling and monitoring techniques. As with precipitation in complex terrain @lundquist_our_2019, the path to more accurate surface wind estimates is through the coupling of multiple data sources and modeling methods.

Traditionally, applications that rely on surface wind analyses, that is, the accurate representation of surface winds on a dense spatial grid, have used numerical simulations from high-resolution atmospheric models @bernhardt_using_2009@mott_meteorological_2010. Numerical simulations, while central to surface wind analysis, face limitations due to their spatial and temporal discretization, leading to structural errors such as the misrepresentation of subgrid processes. Despite advances in improving resolution, physical parameterizations, and data assimilation, these models still struggle with systematic biases and inaccuracies. In addition, these improvements mainly related to increased horizontal resolution typically require a significant increase in computational resources.

Statistical modelling stands as an alternative to numerical simulations that has been used successfully in a wide range of applications in weather and climate. In the context of weather forecasting, a large variety of techniques has been proposed to optimize the information from NWP models, many of which are classified as postprocessing methods #cite(<vannitsem_statistical_2021>). These are typically used to statistically relate NWP model output and other covariates to observations, and obtain bias-corrected and calibrated predictions, with the advantage that they are very cheap to compute. In recent years, neural networks have emerged as a valid option for this task, as they achieve state-of-the-art performance @demaeyer_euppbench_2023 while providing great flexibility: they can accommodate diverse input formats (tabular data, temporal series, images or sequences of images) and allow users to incorporate domain expertise when designing a model’s architecture @zanetta_physics-constrained_2023@dujardin_windtopo_2022.

One of the most important challenges when statistically optimizing NWP predictions is related to the spatiotemporal consistency, that is, preserving the spatial and temporal correlation structure in the postprocessed predictions @vannitsem_statistical_2021. For the many standard approaches that predict univariate probability distributions (for a single variable, at a single location), where each prediction is treated independently, additional computationally involved steps @schefzik_uncertainty_2013@clark_schaake_2004 are required to reestablish correlation structures after sampling. The second crucial challenge is the question of how to blend diverse sources of predictability, such as NWP model outputs and observational data. A key intuition is that both of these challenges are fundamentally related to spatio-temporal covariance modelling, thus making geostatistical methods an attractive option.

A well-established technique traditionally employed in geostatistical modelling is Kriging. It is designed specifically for the purpose of interpolating spatial data. In essence, Kriging makes the assumption that spatial variation in the data can be captured by a statistical model known as a variogram, which quantifies the degree of spatial correlation between data points based on their separation distance. In the realm of machine learning, Kriging is frequently referred to as Gaussian Processes (GPs) regression which can sometimes lead to confusion for researchers transitioning between fields. That said, there are notable distinctions between Kriging and GPs. GPs, which stem from the probabilistic framework, offer more flexible covariance modelling via the use of kernels, expanding beyond the traditional variogram models in Kriging. This flexibility in GPs allows for optimization using maximum likelihood methods, leading to potentially more accurate model predictions @christianson_traditional_2023. Moreover, while Kriging is typically implemented in two or three spatial dimensions, GPs shine in their ability to generalize to higher-dimensional and potentially non-euclidean spaces. In summary, while both Kriging and GPs focus on modelling the covariance between points, the latter presents a broader and more versatile framework, enriched by the vast ecosystem of machine learning @gardner_gpytorch_2021@matthews_gpflow_2017@pinder_gpjax_2022.

In this work, we introduce a novel methodology based on GPs that combines NWP data with additional covariates, from observed wind gust measurements at sparse locations. Recognizing the computational demands of traditional GPs, especially when handling large datasets, our approach implements scalable GP techniques. In particular, we take advantage of Random Fourier Features #cite(<rahimi_random_2007>), an approximation method that significantly reduces computational overhead of the GP. Additionally, we employ the pathwise conditioning of @wilson_pathwise_2021, which facilitates efficient sampling from the Bayesian model. We discuss our results both in a qualitative way, presenting actual realizations of wind fields on a grid, and quantitatively, by evaluating our models on an independent test set.

== Data and methods <4-data_methods>
=== Data <4-data>
This study uses three distinct types of datasets: numerical weather prediction (NWP) model data, digital elevation model (DEM) data, and observational data, all spanning January 2021 to December 2023.

- #strong[NWP]: Our NWP data comes from COSMO-1E hourly analysis — a regional forecasting model that is used operationally in MeteoSwiss. In its high-resolution probabilistic configuration, COSMO-1E runs a 11 member ensemble with a horizontal resolution of 1.1 km and an update cycle of 3 hours. The hourly analysis is obtained via the kilometre-scale ensemble data assimilation system @schraff_kilometrescale_2016, which implements an assimilation scheme based on the ensemble Kalman filter. Note that while some surface wind measurements are assimilated by KENDA, none fall within our study domain. For our use case, we only consider the analysis from the control run. NWP data are bilinearly interpolated on the target points.

- #strong[DEM]: Our DEM has a 50 meters horizontal resolution, from which we derived a set of topographical descriptors known to be related to the dynamics of surface wind, such as the topographical position index, the maximum upwind slope @winstral_spatial_2002 or directional (N-S and E-W) gradients. Data are bilinearly interpolated on the target points.

- #strong[Observations]: Observational data is obtained from a network of more than 500 meteorological stations coming from different sources. These stations are distributed throughout central Europe, with the majority located in Switzerland (see @4-fig-partitioning), and achieve a broad coverage of the region’s diverse weather patterns and local conditions. These observational data comprise hourly wind gust measurements, typically defined as the 1 second average wind maximum. To ensure the consistency and reliability of our results, the observational dataset was carefully quality-controlled. We excluded all stations with anemometers below 6 or above 15 meters height above the ground, as well as obvious measurement errors such as negative wind speeds, or plateau of constant values. Furthermore, a manual validation for suspicious measurements was conducted by cross-checking measurements from adjacent stations, providing a system to identify and correct anomalies.

=== Neural network-based postprocessing baseline <4-baseline>

As a baseline for our results presented in @4-results we use a simple Neural Network Post Processing method (NNPP), similar to the approach presented in @rasp_neural_2018. Here, a neural network is trained to predict the marginal distribution at each prediction point, using the continuous ranked probability score (CRPS) as a loss function. For predictors, the NWP direct model output is used along with other covariates such as topographical descriptors and temporal information (sine and cosine components of the hour of the day). While several ways exist to implement a probabilistic regression for this kind of application, we have chosen one that is as close as possible to the methods presented in the next section, in order to make them more comparable. Therefore, we performed the regression on the transformed data (see @4-appendix-transformation) using the CRPS for the normal distribution as loss function.

=== Surface weather analysis with Gaussian Processes regression <4-swagp>

Let us start by introducing the GP approach to regression, which will be connected later to our problem. Consider a regression task $y = f lr((bold(x))) + epsilon.alt$ where $y in bb(R)$ is the target variable, $x in bb(R)^D$ the inputs and $epsilon.alt tilde.op cal(N) lr((0 , sigma_epsilon.alt^2))$ some additional independent Gaussian observational noise. We assume that $f lr((bold(x)))$ can be described as a Gaussian process, i.e. any finite collection of $f lr((dot.op))$ follows a particular multivariate normal distribution. The process is then entirely defined by a mean function $bold(mu_x) = m lr((bold(x)))$ parameterised by $bold(theta)_m$ and a covariance kernel $bold(K_(x x prime)) = k lr((bold(x) , bold(x) prime))$ parameterised by $bold(theta)_k$ 

$ f lr((dot.op)) tilde.op cal(G P) lr((m lr((dot.op)) , k lr((dot.op , dot.op prime)))).$ <4-eq-gpprior> 

The mean function and the covariance kernel incorporate our knowledge of the data and can be used as a prior distribution from a Bayesian inference perspective. Given a set of $n$ observed locations (or #emph[context] points) $cal(D) = lr((bold(X) , bold(y))) = { bold(x)_i , y_i }_(i = 1)^n$, we can leverage Bayes’ theorem to update $cal(G P)$ and get the posterior distribution conditioned on $cal(D)$, effectively correcting the mean function and reducing the uncertainty of our predictive distribution.

To calculate our GP posterior, we have to consider the resulting joint distribution between the observed locations and the function values at a new target locations $bold(f_(\*)) = f lr((bold(X)_(\*)))$:

$ mat(delim: "[", bold(y);bold(f_(\*))) tilde.op cal(N) lr(
  (mat(delim: "[", bold(mu_x);bold(mu_(\*))) , mat(
    delim: "[",
    bold(K_(x x)) + sigma_epsilon.alt^2 I, bold(K_(x \*));bold(K_(\* x)), bold(K_(\* \*)),

  ))
) $

where $bold(mu_x) = m lr((bold(X)))$ and $bold(mu_(\*)) = m lr((bold(X_(\*))))$ are the mean function values for the observed and target points, respectively, $bold(K_(x x)) = k lr((bold(X) , bold(X)))$ is the covariance matrix, $sigma_epsilon.alt^2$ is the noise variance, and $bold(K_(x \*)) = k lr((bold(X) , bold(X)_(\*)))$ represents the cross-covariance between context and target points. When the joint distribution is a multivariate Gaussian, the marginalisation and conditioning with respect to $bold(y)$ is exact, and the predictive distribution of $f_(\* \| bold(y))$ for target points remains Gaussian and has mean and covariance: 

$   & mu_(\* \| bold(y)) = bold(mu_(\*)) + bold(K_(\* x)) lr([bold(K_(x x)) + sigma_epsilon.alt^2 I])^(- 1) lr((bold(y) - bold(mu_x))) ,\
  & Sigma_(\* \| bold(y)) = bold(K_(\* \*)) - bold(K_(\* x)) lr([bold(K_(x x)) + sigma_n^2 I])^(- 1) bold(K_(x \*)) . $ <4-eq-update>
  
Furthermore, the log marginal likelihood can also be expressed analytically and is given by 

$ log p ( bold(y) lr(
  |bold(X) \) = - 1 / 2 lr((bold(y - mu_x)))^T lr([bold(K_(x x)) + sigma_epsilon.alt^2 I])^(- 1) lr((bold(y - mu_x))) - 1 / 2 log|
) bold(K_(x x)) + sigma_epsilon.alt^2 I \| - n / 2 log 2 pi , $ 

which is used as a loss function during the optimization of a model’s parameters. Let $bold(xi) = { bold(theta)_mu , bold(theta)_k , sigma_epsilon.alt^2 }$ be the set of parameters of our GP, then we use gradient-based optimization on the following objective $ bold(xi)^star.op = "argmin"_(bold(xi) in Xi) - log p lr((bold(y))) thin . $ Now that we have introduced the theoretical background let us consider our application. First, we aim to train a model to use NWP data, topographical and temporal information as inputs $bold(x)$ to predict a surface weather variable $y$ (in our case wind gust), using station measurements as ground truth. Moreover, we are interested in modelling several points $X = x_1 , x_2 , . . . , x_n$ and their joint probability distribution instead of each target location independently, which requires a covariance function $k lr((bold([x,x])))$. 
Combined with a mean function $m lr((bold(x)))$, we have model for our prior probability distribution of $f$ as seen in @4-eq-gpprior. Note that, while the GP model as a whole takes all the input features $bold(x)$, different features are then selected for the mean function or the covariance kernel. For instance, geographical coordinates are part of the inputs but are only used by the kernel.

The following step is then to update the prior probability distribution of $f$ with the observed points from ground stations, represented by $cal(D)$, by applying @4-eq-update. This allows us to obtain probabilistic forecasts that are conditioned on surface measurements. The intuition is quite simple: prior forecasts at target locations will receive an update that is proportional to the covariance with the observed locations, and inversely proportional to $sigma_epsilon.alt^2$. We note that this update can also be viewed as an interpolation of the residuals $y - mu_x$ from the mean function of our prior, and those familiar with geostatistical methods may see that this is conceptually close to regression Kriging. We now proceed to briefly discuss the individual components of our methodology, namely the mean function, covariance function, and observational noise.

==== Mean function <mean-function>
The function $mu : bb(R)^D arrow.r bb(R)$ calculates the average of $f$ for the $D$-dimensional input $bold(x)$. In our application, we show that this function can be implemented as a neural network parameterized by the weights $bold(theta_mu) = bold(w)$. When the NWP forecast is part of $bold(x)$, the mean function can be seen as a bias correction step applied to the NWP forecast. This correction is learned based on information from topographical descriptors and temporal encodings, as well as their interplay. For example, these features indicate whether the target location is sheltered or exposed to the incoming wind. To highlight the added value of the NN-based mean function with respect to simpler alternatives, we present results for multiple configurations in @4-results. There are two approaches to optimize $bold(theta_mu)$: the first involves training these parameters concurrently with $bold(theta_k)$ and $sigma_epsilon.alt^2$, while the second approach entails pretraining the mean function separately and optionally keeping the parameters fixed during the optimization of the posterior GP.

==== Covariance function <covariance-function>
The GP framework provides a highly flexible approach to model covariance between points through a variety of kernel functions, both stationary and non-stationary. In stationary kernels the covariance only depends on the shift between two input values and not the values themselves. A common example of such type of kernels is the radial basis function (RBF) kernel $ k lr((bold(x) , bold(x) prime)) = sigma^2 exp lr((- frac(lr(||) bold(x) - bold(x) prime lr(||)^2, 2 bold(l)^2))) , $ where $l$ and $sigma^2$ are the lengthscale and variance parameters $bold(theta)$ of the kernel. On the other hand, non-stationary kernels allow the output covariance to depend on the input values themselves, and a prominent example is the linear kernel $ k lr((bold(x) , bold(x) prime)) = sigma^2 bold(x) bold(x)^T . $ Kernels can even be combined with neural networks by learning nonlinear transformations of complicated input spaces that map to a simpler latent space where a kernel is applied @wilson_deep_2015. Starting from a base kernel $k lr((bold(x)_i , bold(x)_j divides bold(theta_(b a s e))))$ with parameters $bold(theta_(b a s e))$ the input $bold(x)$ can be transformed as $ k lr((bold(x) , bold(x) prime divides bold(theta_(b a s e)))) arrow.r k lr(
  (g lr((bold(x) , bold(w))) , g lr((bold(x) prime , bold(w))) divides bold(theta_(b a s e)) , bold(w))
) , $ where $g lr((bold(x) , bold(w)))$ is a non-linear mapping given by a neural network parameterized by weights $bold(w)$. Furthermore, multiple kernels can be combined into more expressive ones to account for additive or multiplicative effects of the different factors contributing to the modelled process. Indeed, the sum of two kernels is a kernel, $k lr((bold(x) , bold(x) prime)) = k_1 lr((bold(x) , bold(x) prime)) + k_2 lr((bold(x) , bold(x) prime))$, and a product of two kernels is a kernel, $k lr((bold(x) , bold(x) prime)) = k_1 lr((bold(x) , bold(x) prime)) dot k_2(bold(x) , bold(x)')$ @rasmussen_gaussian_2006.

For the case of wind gust, and spatially distributed phenomena in general, a good first assumption is that the covariance simply depends on distance, and one might consider a kernel with spatial coordinates and elevation as inputs. However, there are cases where the distance alone cannot fully describe the actual covariance between two locations: it might also depend by their specific geomorphological setting or on dynamic factors such as the direction or intensity of the wind, and their interplay with the diurnal cycle. This is where the flexibility of the kernel construction we have just described comes to help: we can design much more complex kernels, capable of including a diverse set of input variables. In @4-results we present three variants of kernel functions.

==== Observational noise <observational-noise>
As the name suggests, the observational noise represents the uncertainty of the observed measurement $y$. It is analog to the #emph[nugget] parameter in the context of Kriging. When accounted for, it can make the GP model more robust to outliers and noise in the data. This is because the noise term in the GP model acknowledges that the observed data might not be perfectly accurate. Furthermore, the presence of noise influences the model’s predictive uncertainty. In areas where the model has observed (noisy) data $y$, it will still have some uncertainty, and the function $f_(\| y)$ will not be exactly $y$. This also has an effect on the \"smoothness\" of the predictions: by accounting for some noise, $f_(\| y)$ will vary more smoothly around the observed data. Importantly, observational noise requires scaling due to the nonlinear transformation performed on the target data distribution (to make it standard normal), as explained in @4-appendix-transformation. The observational noise parameter can either be learned during optimization or be a fixed value, arbitrarily chosen by the user based on the importance they give to observed data. In our experiments, we have observed that keeping the observational noise constant led to a more stable training. For this reason, we have chosen to fix the observational noise $sigma_epsilon.alt^2$ to 1 $m s^(- 1)$ (in the original, untransformed space).

=== Problem definition, training and evaluation setup <problem-definition-training-and-evaluation-setup>
In defining the problem, we start with the simplifying assumption that data from different timesteps $t$ are independent. In other words, we only focus on modelling spatial dependencies and do not explicitly consider temporal ones. In a single time step $t$, our data consists of $n$ spatially distributed input points ${ bold(x)_1 , bold(x)_2 , . . . , bold(x)_n } in bb(R)^D$ and corresponding outputs ${ y_1 , y_2 , . . . , y_n } in bb(R)$. For simplicity, we denote them as $arrow(X) in bb(R)^(n times D)$ and $arrow(Y) in bb(R)^n$, and we denote $cal(D)_t = lr((arrow(X)_t , arrow(Y)_t))$ as the set of points at time $t$. A single timestep therefore represents a single GP regression task: we are not interested in modelling the covariance across different timesteps but only the spatial covariance.

Before we discuss our training strategy, let us describe the data partitioning method. Our dataset is split into training, validation and test sets. We used 70% of the stations and 2 years of data for training, 10% of stations and 1 year for validation and 20% and 1 year for testing. For the validation and test sets, we want to evaluate the performance of the model at a set $cal(T)$ of #emph[target] stations given a set $cal(C)$ of observed #emph[context] stations. Therefore, we represent an evaluation task as $cal(D)^(e v a l) = { lr((arrow(X)^t , arrow(Y)^t)) , lr((arrow(X)^c , arrow(Y)^c)) } = { cal(T) , cal(C) }$. Note that context stations are points that the model has seen during training, which are used to predict on new locations (targets) during evaluation. Further details on the partitioning method are provided in @4-appendix-partitioning. After splitting our dataset, we standardize our input using the mean and standard deviation of the training set, and more importantly, we also transform the output values to a standard normal distribution (see @4-appendix-transformation for details).

In terms of training strategy, a naïve approach would be to fit a different GP model to each individual timestep. However, it has the inconvenience that the optimization procedure must be repeated several times independently – which is a computational burden, leads to varying results, and is potentially more prone to overfitting due to the small number of points. A more robust approach is to optimize a model that achieves good performance across many tasks. This can be done by considering a metalearning approach, where the training loop is applied on tasks. Our method is similar to the one described by @patacchiola_bayesian_2020, with the difference that at each step we update the model parameters using mini-batches of tasks instead of a single task. The pseudocode is given in Algorithm
// ~#link(<alg:training>)[\[alg:training\]].
Let $bold(cal(D)) = { cal(D)_t }_(t = 1)^T$ be the complete training data set consisting of $T$ tasks and $bold(xi)$ the set of model parameters. At each training step, we sample a batch of tasks $cal(D)_i tilde.op bold(cal(D))$, and for all tasks we compute the negative log marginal likelihood using Eq.
// ~#link(<eq:mll>)[\[eq:mll\]]
. We then take the average of the loss for the entire batch and use it to perform the parameters update based on gradient descent. During training, we also computed the loss in a validation set. More details about the training procedure are presented in @4-appendix-modelling.




#algorithm(
  caption: [Optimizing GP model parameters across all tasks],
  pseudocode(
    no-number,
    [*Require:* train dataset $cal(D) = {cal(D)_t}_(t=1)^T$],
    no-number,
    [*Require:* model parameters $bold(xi)$],
    [initialize model parameters],
    [*while* not done *do*],
    ind, [sample batch of N tasks $cal(D)_i tilde bold(cal(D))$],
    [*for all* $cal(D)_i$ *do*],
    ind, [compute $cal(L)_i = -"log" p(cal(D)_i, bold(xi))$],
    ded, [*end for*],
    [compute $cal(L) = 1/N sum_(i=1)^N cal(L)_i$],
    [update $bold(xi) arrow.l bold(xi) alpha nabla_(bold(xi))cal(L)$],
    ded, [*end while*]
  ),
  placement: top,
)

=== Scalability to large grids <scalability-to-large-grids>
A critical aspect concerning GPs is that they typically do not scale well with the amount of data. Specifically, the exact computation requirements of the GP algorithm scale cubically in the number of training points, and storage requirements (for the covariance matrix) follow a quadratic increase in the number of training points. Fortunately, recent work has provided a variety of new approximation methods that greatly improve the computational efficiency of GPs, including Sparse GPs, Sparse Variational GPs, others.

In addition to the poor scaling of exact GPs in the number of training points, similar limitations are found when building realizations of the model across large sets of target points. In the context of spatiotemporal modelling this can become quite challenging as predictions are often made on a dense grid. To grasp the scale of the numbers involved, consider a target grid of 1000x1000 pixels, with $n = 1 prime 000 prime 000$ input points $bold(X) = { bold(x)_1 , bold(x)_2 , . . . , bold(x)_n }$. To obtain the full multivariate distribution, one would need to compute a covariance matrix $K lr((bold(X) , bold(X)))$ of size $n^2$. When working with double precision floats, which is often the case in the context of GPs due to numerical stability, this would require to store an 8-#emph[terabyte] large matrix. For all practical purposes, this is intractable. A viable solution to this problem is to avoid computing the full covariance matrix altogether via random Fourier feature methods.

==== Random Fourier features <sec:rff>
One effective strategy to address the scalability challenges in Gaussian Processes (GPs) is the adoption of Random Fourier Features (RFF). This technique allows for the approximation of shift-invariant kernel functions, making it especially advantageous for GPs employing commonly used kernels such as the Radial Basis Function (RBF) or the Matern family kernels. Given a shift-invariant kernel $k lr((bold(x) , bold(x prime))) = k lr((lr(||) bold(x) - bold(x prime) lr(||)))$ where $bold(x)$ and $bold(x prime)$ are input vectors, RFF provides an approximation $ k lr((bold(x) , bold(x prime))) approx bold(phi.alt) lr((bold(x)))^tack.b bold(phi.alt) lr((bold(x prime))) . $ The kernel can be understood as an inner product in an infinite-dimensional feature space approximated by a finite number of Fourier components $L$, and $bold(phi.alt) : cal(X) arrow.r bb(R)^L$ is the feature map to the $L$-dimensional approximation of the said feature space. The feature map is defined as

$
bold(phi.alt) lr((bold("x"))) := sqrt(2 / L) mat(
  delim: "[",
  sin lr((bold(omega)_1^tack.b bold(x)));cos lr((bold(omega)_1^tack.b bold(x)));dots.v;sin lr((bold(omega)_(L \/ 2)^tack.b bold(x)));cos lr((bold(omega)_(L \/ 2)^tack.b bold(x))),

) , quad bold(omega)_i tilde.op^(i i d) P lr((omega)) , 
$ <4-eq-featuremap>

with $P lr((omega))$ denoting the spectral density of the kernel (interpreted as a probability distribution over frequencies) and $L$ the number of random features. For instance, the spectral density $P lr((omega))$ of the RBF kernel is the Gaussian distribution. Then, it is possible to define a random function of a GP whose covariance is approximately $k lr((dot.op , dot.op))$

$
f lr((dot.op)) approx mu lr((dot.op)) + bold(phi.alt) lr((dot.op))^tack.b bold(w) = mu lr((dot.op)) + sum_(i = 1)^L w_i phi.alt_i lr((dot.op)) , quad w_i tilde.op cal(N) lr((0 , 1)) thin .
$

Importantly, this formulation allows us to draw realizations from an approximate GP prior in linear complexity with the number of inference points, without the need to compute the covariance matrix. In practice, we have a function that can be realized anywhere in our input space, where the stochasticity is entirely controlled by $bold(w)$. This also means that during inference the computation of a realization can be divided into chunks, to further reduce memory usage, and since $bold(w)$ is kept constant the aggregated chunks will still yield a coherent sample, without any artifact at the chunks boundaries.

As shown in the previous section, it is possible to combine multiple kernels via summation or multiplication. In our study we consider the product combination for our kernels, and this can also be implemented when using the RFF approximation. Consider two shift-invariant kernels $k_1$ and $k_2$. Their respective feature maps are defined following @4-eq-featuremap. We obtain the product of the two kernels by combining the sampled frequencies $arrow(omega)_1$ and $arrow(omega)_2$ as $ arrow(omega)_(upright("combined")) = sqrt(arrow(omega)_1^2 + arrow(omega)_2^2) , $

and subsequently apply @4-eq-featuremap.

==== Pathwise conditioning <4-pathwise_conditioning>
Gaussian posteriors are typically obtained via a location-scale transform applied on mean vectors and covariance matrices of the prior distribution. The avenue for prior sampling offered by RFF allows us to express Gaussian posteriors in a different but equally valid way, which focuses on samples (or #emph[paths]) instead of distributions. Such an approach is presented in @wilson_pathwise_2021. To compute a posterior sample $lr((f divides bold(y))) lr((dot.op))$, a two steps procedure is required: first, a prior sample at the target location $f lr((dot.op))$ is generated, and the same sample is also evaluated at the observed location $f lr((X))$. Then, an update term is computed based on the difference between the $f lr((X))$ and the observed value, and it is then added to the prior to yield the posterior sample conditioned on the observed data. This procedure is described by

$ underbrace(lr((f divides bold(y))) lr((dot.op)), upright("conditional ")) =^(upright(d)) underbrace(f lr((dot.op)), upright("prior ")) + underbrace(
  K lr((dot.op , bold(X))) K lr((bold(X) , bold(X)))^(- 1) lr((bold(y) - f lr((bold(X))))),
  upright("update "),

) , $

approximated by

$ lr((f divides bold(y))) lr((dot.op)) approx sum_(i = 1)^L w_i phi.alt_i lr((dot.op)) + sum_(j = 1)^n v_j k lr((dot.op , bold(x)_j)) , $

where $bold(v) = bold(K)_(n , n)^(- 1) lr((bold(y) - bold(Phi) bold(w)))$. The great advantage of this way of obtaining posterior samples is that it still scales linearly in the number of target points while allowing splitting computation into chunks.

#figure(
  image("scorecard.png"),
  caption: [
    Scorecard summarizing the evaluation metrics on the test dataset. Values in each box indicate the thresholded (at 4 m/s) CRPS, and colors indicate the relative performance with respect to the baseline, as shown in the colorbar.
  ],
  placement: top,
) <4-fig-scorecard>


== Experiments and results <4-results>
We have designed an experimental setup with the goal of presenting results for different modelling choices, degrees of approximation, and type of predictions.

Differences in modelling choice concern how the mean function and the covariance function are defined. This can be highly flexible, allowing users to adapt their model’s complexity or to account for domain knowledge instead of purely relying on data-driven optimization. Although a large variety of solutions is possible, we present here only a few combinations that allow us to draw some general conclusions. By degrees of approximation we mean the number of Fourier features used in the RFF approximation of the covariance kernel. The type of prediction is a distinction between using the prior predictive distribution (without conditioning on the observed data), the posterior predictive distribution (with conditioning on context, nearby stations) and the #emph[hyperlocal] predictive distribution, which is the prediction at locations where the observation itself is available and used.

These results, all based on the test dataset $cal(T)_(t e s t)$, are summarised with a scorecard in @4-fig-scorecard in terms of CRPS (with threshold at 4 $m s^(- 1)$) and its relative skill, color-coded, compared to the plain neural-network postprocessing (NNPP) approach chosen as a baseline. For the NNPP approach we have a CRPS of 0.87 $m s^(- 1)$ Additionally, we evaluate the calibration of our models by the aid of conditional PIT histograms and reliability diagrams (for the probability of exceeding thresholds of 10 $m s^(- 1)$, 20 $m s^(- 1)$ and 30 $m s^(- 1)$). More details on the evaluation metric and diagnostic tools can be found in @4-appendix-evaluation.

=== Mean modelling <4-mfexp>
#block[
  #box(width: 100%, )

]

#figure(
  image("mean_exp_reliability_diagrams.png"),
  caption: [
     Reliability diagrams for the mean modelling experiments, with columns representing the different exceedance thresholds (10, 20 and 30 m/s) and the first and second rows are for prior and posterior predictions respectively. The secondary axis shows the log-scaled number of occurrences for each bin.
  ],
  placement: top,
) <4-fig-mean_exp_rel>



#figure(
  image("mean_exp_pit.png"),
  caption: [
    PIT histograms for the mean modelling experiments, with the results for the prior and posterior predictions on the left and right hand side respectively. Perfect calibration would be indicated by a uniform histogram.
  ]
) <4-fig-mean_exp_pit>

For the mean function, we have considered three approaches in which increasingly more information is added to the prior: a learned constant value, a mean function based on the raw COSMO-1E wind gust, and a mean function based on the bias corrected (via a neural network) COSMO-1E wind gust. In all cases, the same kernel function $k_(S D)$, described below, was used. It is our purpose here to demonstrate the effect of including more prior information whenever it is available, since it is common in the GP literature to prioritize the modelling of the covariance rather than the mean function of the prior. In @4-fig-scorecard, the considered approaches are on the first, second and fourth row respectively. Focusing on the exact GP, without RFF approximation, we observe that the neural mean approach significantly outperforms the constant mean and NWP mean approaches, and is the only approach that, for posterior predictions, shows positive skill compared to the NNPP baseline.

@4-fig-mean_exp_rel shows the reliability diagrams for the models considered in this experiment. In the top row, the diagrams for the prior predictions show two interesting aspects: first, the constant mean approach performs poorly, as it barely predicts any probability above 10%. This is to be expected because the model simply learned a global climatology and does not exhibit any sharpness, therefore missing out large values completely. The NWP mean approach, on the other hand, overpredicts wind gust for all thresholds. This is consistent with the error structure of COSMO-1E itself, which tends to overestimate the wind gust, especially in complex topography. Prior predictions for the neural mean approach are well calibrated for the 10 m/s threshold, but underestimate probabilities for the other thresholds. @4-fig-mean_exp_pit shows the PIT histogram for the same experiment, and the results for the prior, on the left-hand side, confirm what we have observed in the reliability diagrams: the constant mean approach completely misses large values, as indicated by the increased density in the right-most bin; the NWP mean approach has a positive bias, as indicated by the slope of the PIT histogram; the neural mean approach has the best overall calibration, but still shows a slight negative bias.

When considering the posterior predictions, as shown in the bottom row of @4-fig-mean_exp_rel, we see a clear improvement for all approaches. As expected, the largest change is observed for the constant mean model, since it has learned to rely almost entirely on observed data. We also observe a clear overestimation of the probabilities for the 20 m/s and 30 m/s thresholds. For the NWP mean model, we see a general reduction in the overestimation. For the neural mean approach, we see a general improvement in the calibration, especially for the 20 m/s and 30 m/s thresholds. The right-hand panel in @4-fig-mean_exp_pit also reflects the observed changes.

These results highlight the importance of providing an appropriate mean function to the model’s prior. We have observed that, in our problem, including NWP information is crucial to obtain results comparable or better than the baseline, and that it is also important that the provided mean function is unbiased, thus requiring a sort of postprocessing step applied to the raw NWP information. We note that while these results are true for the case of wind modelling, different outcomes could be obtained in other applications like temperature, where the observations are much more representative of the surroundings and an appropriate prior mean is therefore of less importance.

#figure(
  image("cov_exp_reliability_diagrams.png"),
  caption: [
     Reliability diagrams for the covariance modelling experiments, with columns representing the different exceedance thresholds (10, 20 and 30 m/s) and the first and second rows are for prior and posterior predictions respectively. The secondary axis shows the log-scaled number of occurrences for each bin.
  ]
) <4-fig-cov_exp_rel>

#figure(
  image("cov_exp_pit.png"),
  caption: [
    PIT histograms for the covariance modelling experiments, with the results for the prior and posterior predictions on the left and right hand side respectively. Perfect calibration would be indicated by a uniform histogram.
  ]
) <4-fig-cov_exp_pit>


=== Covariance modelling <4-covexp>
For the covariance function, we experimented with three variants of increasing complexity. We start with a simple spatial kernel $k_S : bb(R)^3 arrow.r bb(R)$ as an RBF kernel taking as inputs easting and northing coordinates and elevation. We parameterize it as an automatic relevance determination kernel, which means that we have one lengthscale for each input dimension. In practice, this means that some dimensions will be more important than others. We let the model learn both $bold(l) in bb(R)^3$ and the variance parameter $sigma^2$ (starting with some sensible defaults).

The second variant consists of the product combination of the spatial kernel and a deep kernel, $k_(S D) lr((x , x prime)) = k_S lr((x , x prime)) dot.op k_D lr((x , x prime))$. Their combination was motivated by the shortcomings of the individual components. While the spatial kernel $k_S$ is likely too simplistic, a deep kernel can be susceptible to overfitting by wrongly learning spurious correlations over vast distances. Therefore, the covariance is forced to zero after a certain distance, preventing the model from picking up spurious correlations while still learning useful nonlinear relationships. We note that one could take advantage of this by using sparse computations as shown in @furrer_covariance_2006. However, this was not considered in this study.

The third variant $k_(S D) lr((x , x prime)) = k_S lr((x , x prime)) dot.op k_D lr((x , x prime)) dot.op k_L lr((x , x prime))$ includes a linear scaling component on top of the spatial deep kernel. This allows us to learn a non-stationary component of the covariance structure, specifically a scaling term that depends on $x$ and $x prime$ and not solely on their distance.

For all variants, the same mean function is used, namely the neural mean function.

The models considered for this experiment are shown in the third, fourth and fifth row of @4-fig-scorecard. For prior predictions, we observe no significant difference in performance, with the exception of a small improvement for the $k_(S D L)$ kernel, which is on par with the baseline if no approximation is used. The difference between $k_(S D L)$ and the other two kernels that perform equally is due to the fact that $k_(S D L)$ is nonstationary, and is therefore able to learn an input-dependent prior marginal variance. On the other hand, stationary kernels have a fixed marginal variance for the prior predictive distribution.

@4-fig-cov_exp_rel shows realiability diagrams for the prior and posterior prediction in the top and bottom row respectively. In general, results appear quite similar between the different models. We observe a general tendency to underestimate probabilities for prior predictions, whereas the opposite happens for posteriors although they are closer to being well calibrated. We should note that the number of occurrences for the second and third thresholds is very small for large probabilities, which makes the observed miscalibration less significant. @4-fig-cov_exp_pit confirms the same tendency: prior predictions are negatively biased and posteriors are not.

We observe that the largest improvement comes from the combination of the spatial kernel with the deep kernel, whereas the addition of the linear scaling does not change our results in a significant way.

==== Approximation error <4-approximation_error>
@4-fig-scorecard also shows the effect of the RFF approximation on the quality of the predictions. In general, we observe that only posterior predictions suffer significantly from the approximation, whereas almost no effect is observed for prior predictions. We believe this might be explained by the fact that the prior function is smoother than the posterior and is therefore easier to approximate. As expected, increasing the number of Fourier features reduces the approximation error, with the trade-off of more expensive computation.

== Case study <4-case_study>
A qualitative analysis of the forecasts for a significant event is now provided. The predictions are generated by a model that integrates the Neural mean and the Spatial-Deep-Linear kernel, leveraging an RFF approximated kernel with 2048 Fourier features.

We examine the event that took place between March 31 and April 1, 2023, coinciding with the passage of storm Mathis across Switzerland. Our focus is on the northeastern part of Switzerland, which bore the brunt of the storm’s impact, and we present our findings on a grid with a horizontal resolution of 250 meters. @4-fig-mathis_maximum illustrates the average predictions and standard deviation of the maximum wind gust predicted by each member of the ensemble during the event, alongside the observed values represented by circles. Overall, there appears to be strong agreement between the forecasts and the actual observations, displaying a realistic spatial pattern. The spatial pattern of the standard deviation is influenced by the distance from the measurement stations, particularly in flat regions. The explanation becomes more complex in areas with varied topography, where factors such as elevation variations and other geomorphological characteristics play an important role.

#figure(
  image("mathis_nw_switzerland_gust_maxmean_maxstd.png"),
  caption: [
    *a.* The ensemble mean of the Mathis storm event’s maximum wind speed, along with the measured station values shown in circles. *b.* The ensemble standard deviation of the Mathis storm event’s maximum wind speed, with the location of the measuring stations shown with circles.
  ]
) <4-fig-mathis_maximum>

@4-fig-koppigen shows the meteograms for the Koppigen measuring station, where the maximum wind gust reached 136 km/h during the event, marking the highest value recorded in the area. Similarly to the diagram in @4-fig-scorecard, we present results for three forecast categories: prior, posterior, and hyperlocal predictive distribution. Contrasted with the prior predictions, the posterior predictive distribution seems narrower and somewhat more precise, indicating the beneficial impact of the information sourced from neighboring stations. Nevertheless, it is evident that certain high-intensity peaks are overlooked, possibly due to their highly localized nature and related unpredictability. Naturally, when examining the hyperlocal predictive distribution, these peaks are accurately represented.
#figure(
  image("KOP_wind_gusts.png"),
  caption: [
    Meteogram of the predicted and observed wind gust at the Koppigen station (KOP) during the Mathis storm event, using the approximated RFF covariance. While the prior prediction (above) misses the most intense.
  ]
) <4-fig-koppigen>

The RFF approximation, in combination with the pathwise conditioning approach, allows us to efficiently sample from the posterior predictive distribution. @4-fig-realizations shows a series of random realizations at a single timestep during the event. All realizations have a realistic spatial structure coherent with the used topographical features. However, it is important to consider that this covariance structure was learned in a data-driven way and from sparse observations, and the learned covariance structure was approximated. While these realizations certainly look plausible, there is no guarantee that they are physically coherent.

#figure(
  image("mathis_nw_switzerland_realizations.png"),
  caption: [
    A set of generated realizations from the posterior predictive distribution for the north-west region of Switzerland, during the Mathis storm event. Each realization displays a realistic spatial structure while still providing a good degree of variability on several spatial scales.
  ]
) <4-fig-realizations>

#figure(
  block(inset: (right: -1.8cm), clip: true, width: auto, image("mathis_inca.png")),
  caption: [
    A single generated realization from the posterior predictive distribution for the Swiss radar domain, during the Mathis storm event. The generated field displays a realistic spatial structure. The field covers a domain spanning 710km in longitude and 640km in latitude, and with a resolution of 250 meters it totals approximately 7.2 million pixels. Note that the real size of the grid cells is smaller than pixels in the image, resulting in some higher values being barely visible to the eye.
  ]
) <4-fig-mathis_inca>

== Conclusions and outlook <4-conclusions>
This study presented an innovative methodology for accurately representing surface wind gusts through a combination of machine learning methods, namely GPs and neural networks. We have used these methods to successfully incorporate information from NWP models with information available in the form of ground observations. We have shown the added value of including surface measurements compared to simply applying a postprocessing model to NWP forecasts. We addressed the computational challenges of GPs by leveraging state-of-the-art techniques for efficient sampling, which allows our models to generate spatially coherent realizations of wind fields that are also marginally calibrated.

The experiments conducted have showcased the flexibility and robustness of our approach. By evaluating different model configurations and the impact of approximation techniques, we have identified optimal strategies for balancing computational efficiency, complexity and predictive accuracy. Based on our results, we can conclude that our case study on the storm event further exemplifies the practical relevance and applicability of our methodology in capturing and forecasting significant weather phenomena with high precision.

We envisage that the following aspects could be further explored in future research:

- Modelling the temporal covariance structure explicitely, by including a temporal covariance kernel function. This could bring potentially significant benefits when considering a finer temporal granularity, for instance 10 minutes instead of hourly granularity as presented here;

- Explore the potential of the proposed methodology for nowcasting applications by developing an appropriate blending function controlled by the observational noise;

- Further improving the data transformation technique. For example, one might consider learning input-dependent transformations as proposed in @maronas_transforming_2021;

- Extending current models to handle vector data, such as wind vectors.

Furthermore, an interesting question is the comparison of the proposed methodology with similarly scoped approaches such as neural processes, for instance, the ConvGNP model architecture proposed in @andersson_environmental_2023, or the MetNet-3 model by @andrychowicz_deep_2023.



== Appendix <4-appendix>

=== Data partitioning <4-appendix-partitioning>
For partitioning the stations, we used stratified random sampling, by which the stations were first grouped by their 99$""^(t h)$ quantile values. This was an easy-to-implement approach to ensure that heterogeneous conditions in the surface stations (hyperlocal conditions, type of measuring device, height of measuring device, etc.) were equally represented in the different sets, with a focus on the tails of the distributions. The time partitioning is sequential, with years 2020 and 2021 for the training set, 2022 for validation, and 2023 for the test set. As explained in the main text, the evaluation of the validation and test set is performed on a set $cal(T)$ of station measurements, with other sets of station measurements $cal(C)$ used as context. In the case of validation, stations from the training set $bold(cal(D))$ are used as context. In the case of testing, stations from both training $bold(cal(D))$ and validation targets $bold(cal(T)_(v a l))$ are used as context. A visual representation of the partitioning strategy is shown in #ref(<4-fig-partitioning>)a. #ref(<4-fig-partitioning>)b shows the spatial distribution of $bold(cal(D))$, $bold(cal(T)_(v a l))$ and $bold(cal(T)_(t e s t))$. While this partitioning strategy does not create strictly independent datasets, we believe it is the best possible solution. It reflects the real-world conditions in which the presented methodology would be applied, where the stations used during training will be used as context for the predicted fields. Moreover, the same approach was already used in related work by @scholz_sim2real_2023.

#figure(
  image("partitioning.png"),
  caption: [
    *a*. Schematic of the data partitioning strategy. *b*. Spatial distribution of the $cal(D)$, $cal(T)_("val")$ and $cal(T)_"test"$ sets.
  ]
) <4-fig-partitioning>

=== Data transformation <4-appendix-transformation>
One of the basic assumptions of GPs is that the random variable of interest is normally distributed. In real-world applications, this is often not the case, and the use of GPs with non-conjugate posteriors is an active area of research. While some theoretically principled approaches exist to deal with these situations, such as Markov Chain Monte Carlo methods, Laplace approximation, or Variational Inference, we opted for a simpler approach and tested it empirically on our problem. Our approach consists of transforming the original data to a Gaussian distribution using a parametric transformation that is also bijective and differentiable.

We have defined a parametric transformation for our data of the form $ z = - frac(log a / y - c, b) $ with its corresponding inverse transformation $ y = frac(a, c + e^(- b z)) $ where parameters $a = 4.66$ and $b = 0.74$ and $c = 0.08$ were found via curve fitting on a corresponding empirical quantile transformation. A Jupyter notebook illustrating the procedure is included in the code repository. Compared to using the quantile transformation directly, we have observed that this parametric version behaved more robustly, especially towards the right tail of the distribution where there is a very low density of points. Another advantage of this parametric transformation is that it allows us to easily compute its derivative, $frac(d z, d y) = frac(a, b y lr((a - c y)))$, which can be used to scale the observational noise to compensate for the deformation of the density distribution. On the other hand, using a homoscedastic observational noise in the transformed space would result in a heteroscedastic observational noise in the original space, especially for large values of $y$, after applying the inverse transformation to sampled values.

=== Evaluation metrics and tools <4-appendix-evaluation>
We evaluate the performance of our models using standard tools for probabilistic forecast verification. Specifically, we have used the CRPS, as defined by $ upright("CRPS") lr((F , y)) = integral_(bb(R)) lr([F lr((x)) - bb(1) { y lt.eq x }])^2 d x , $ to evaluate sharpness and calibration of the forecasts. To put less emphasis on small values of wind gusts, which are practically irrelevant for our application, we have decided to opt for a threshold-weighted version of the CRPS @allen_weighted_2022, and considered the twCRPS for a threshold value of 4 $m s^(- 1)$. We have also used two other tools to evaluate calibration. The reliability diagram, which shows how well the predicted probabilities of an event correspond to their observed frequencies, can be used to assess calibration with respect to specific thresholds. The probability integral transform (PIT) histogram @gneiting_probabilistic_2007, can be used to assess the overall calibration and determines whether the random variable of interest is indeed sampled from the predicted distribution. As with the twCRPS, we opted for a version of the PIT histogram, the conditional PIT @allen_weighted_2022, that emphasizes calibration above the same threshold.

=== Modelling details <4-appendix-modelling>
==== Architectures, optimization and predictors

The neural network used for the mean function was a fully connected network with 2 layers and 32 units each, using the hyperbolic tangent activation function. The same architecture was used for the deep kernel $k_D$, with the only difference that it outputs a two-dimensional vector (which is passed to the base kernel) instead of a one-dimensional vector. All models were trained with a batch size of 64 timesteps/tasks and used the Adabelief optimizer @zhuang_adabelief_2020 with a learning rate of $10^(- 4)$. During training, the repeated presence of the same set of stations at all steps could potentially lead to co-adaptation/overfitting. To mitigate this risk, each sampled task contained a different random set of 128 stations. Additionally, the CRPS for marginal predictions was computed as a validation loss and used to diagnose overfitting to the training dataset and guided us in the choice of the models’ hyperparameters. Additional details about the models can be inspected in the manuscript’s accompanying GitHub repository.

#figure(align(center)[#table(
      columns: 2,
      align: (col, row) => (right, right,).at(col),
      inset: 6pt,
      [Name],
      [Type],
      [Surface wind speed of gust],
      [Dynamic],
      [Surface wind speed of gust ($t - 1$ hour)],
      [Dynamic],
      [Surface wind speed of gust ($t + 1$ hour)],
      [Dynamic],
      [Change in wind speed of gust ($t - 1$ to $t + 1$)],
      [Dynamic],
      [Sx @winstral_spatial_2002],
      [Dynamic/Static],
      [Sine component of the hour of the day],
      [Temporal],
      [Cosine component of the hour of the day],
      [Temporal],
      [TPI (500 m scale)],
      [Static],
      [TPI (2000 m scale)],
      [Static],
      [Model elevation difference],
      [Static],
      [West-East derivative (2000  scale)],
      [Static],
      [South-North derivative (2000  scale)],
      [Static],
    )], 
    caption: [Predictors used by the NNPP baseline model.]
) <4-tab-nnpp_predictors>

#figure(align(center)[#table(
      columns: 2,
      align: (col, row) => (right, right,).at(col),
      inset: 6pt,
      [Name],
      [Type],
      [Surface wind speed of gust],
      [Dynamic],
      [Surface wind speed of gust ($t - 1$ hour)],
      [Dynamic],
      [Surface wind speed of gust ($t + 1$ hour)],
      [Dynamic],
      [Sx @winstral_spatial_2002],
      [Dynamic/Static],
      [Sine component of the hour of the day],
      [Temporal],
      [Cosine component of the hour of the day],
      [Temporal],
      [TPI (500 m scale)],
      [Static],
      [TPI (2000 m scale)],
      [Static],
      [Model elevation difference],
      [Static],
      [West-East derivative (2000  scale)],
      [Static],
      [South-North derivative (2000  scale)],
      [Static],
    )], 
    caption: [Predictors used by the neural mean.]
) <4-tab-neural_mean_predictors>

#figure(align(center)[#table(
      columns: 2,
      align: (col, row) => (right, right,).at(col),
      inset: 6pt,
      [Name],
      [Type],
      [Elevation],
      [Static],
      [Easting in Swiss coordinates],
      [Static],
      [Northing in Swiss coordinates],
      [Static],
    )], caption: [Predictors used by the spatial kernel.]
) <4-tab-spatial_kernel_predictors>

  #figure(align(center)[#table(
      columns: 2,
      align: (col, row) => (right, right,).at(col),
      inset: 6pt,
      [Name],
      [Type],
      [Eastward wind component],
      [Dynamic],
      [Northward wind component],
      [Dynamic],
      [Model elevation difference],
      [Static],
      [Elevation],
      [Static],
      [Easting in Swiss coordinates],
      [Static],
      [Northing in Swiss coordinates],
      [Static],
      [West-East derivative (2000  scale)],
      [Static],
      [South-North derivative (2000  scale)],
      [Static],
    )], caption: [Predictors used by the deep kernel.]
) <4-tab-deep_kernel_predictors>

#figure(
    align(center)[#table(
        columns: 2,
        align: (col, row) => (right, right,).at(col),
        inset: 6pt,
        [Name],
        [Type],
        [Surface wind speed of gust],
        [Dynamic],
        [Change in wind speed of gust ($t - 1$ to $t + 1$)],
        [Dynamic],
        [Sine component of the hour of the day],
        [Temporal],
        [Cosine component of the hour of the day],
        [Temporal],
        [Model elevation difference],
        [Static],
        [TPI (2000 m scale)],
        [Static],
        [West-East derivative (2000  scale)],
        [Static],
        [South-North derivative (2000  scale)],
        [Static],
      )],
    caption: [Predictors used by the linear scaling component of the Spatial-Deep-Linear kernel.],
) <4-tab-linear_kernel_predictors>

==== Parameters constraints <4-appendix_constraints>
When training GPs models, one must take into consideration that some parameters can only take positive values, such as a kernel’s variance and lengthscale parameters or the observational noise. To ensure that these parameters do not become negative during optimization, we have considered two common approaches. The first approach was the same implemented in @pinder_gpjax_2022, which uses bijective transformations to map parameters to and from an unconstrained space where the gradient updates are applied. This approach is very general and can accommodate a large variety of constraints over the model parameters. Since we were only interested in ensuring that parameters remain strictly positive, we opted for another approach that simply consists of modifying the optimization updates to keep parameters above zero. The constraint function $f$ is then defined as $ f lr((p , u)) = cases(
  delim: "{",
  - p + delta & upright("if ") p + u < delta,
  u & upright("otherwise"),

) $ where $p$ is the current value of a parameter, $u$ is the computed update (already scaled by the learning rate) based on the gradient of the loss function, and $delta$ is a small positive value.

=== Libraries and computational resources <4-appendix_libraries>
Our approach centers around JAX @bradbury_jax_2018, a framework designed for high-performance computing and machine learning, which includes automatic differentiation features. We incorporated modified code from the GPJax @pinder_gpjax_2022 library. We utilized a single NVIDIA Tesla A-100 GPU for model training, significantly reducing the training time, thanks to JAX’s efficient just-in-time compilation. For instance, training the model with the neural mean and the spatial deep kernel took us around one minute. We conducted our computations at the Swiss Centre for Scientific Computing (CSCS).


