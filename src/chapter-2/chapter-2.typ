#import "@preview/cetz:0.2.2"
#import "../utils.typ": *

= Weather information optimization toolbox <chap-2>
This chapter outlines the key methodological bases that underpin the statistical optimization of weather information. Firstly, we will explore forecast verification in @2-forecast_verification. We will understand how to assess the quality of a forecast, which is crucial to ensure our optimization objectives are set correctly. Next, in @2-feature_selection, we will explore the selection and creation of relevant predictive features, which is the stage in any machine learning application that requires the most domain expertise. Finally, in @2-machine_learning, we will provide an easy-to-understand introduction to important concepts in machine learning that are particularly useful for our objectives, to assist readers who are not already familiar with these methods. In summary, we will learn that achieving success in optimizing weather information requires both meteorological expertise and a solid grounding in data science and statistical learning. 

#pagebreak()

== Forecast verification <2-forecast_verification>

A forecast has no intrinsic value if it is not accompanied by an understanding of how well it performs. Therefore, verifying the quality of forecasts is just as important as issuing them. Doing so allows to gain awareness of the forecasts' strengths and limitations, so that forecasters learn how forecasts can be improved and users may better support their decision making process @jolliffe_forecast_2011@wilks_chapter_2019. 

#citet[@murphy_what_1993] presents differing views of what constitutes a good forecast and discusses three main aspects: _consistency_ between the forecaster's judgement and the forecast, _quality_ in terms of the correspondence between forecasts and observations and _value_ in terms of the amount of information that a forecast provides to benefit the users. Objective ways to evaluate the last two aspects exist, and we will focus on those in the remainder of this section. Generally, they involve the quantitative comparison between forecasts and the corresponding observations that materialize. 

From the perspective of product developers, several reasons motivate conducting an objective forecast verification. As already mentioned, it can be primarily used to improve forecasting systems. Evaluating forecasts with the goal of understanding their strengths and weaknesses is regarded by #citet[@murphy_what_1993] as _diagnostic_ verification. It relies on the analysis of verification statistics, by understanding different components of the performance of a forecasting system (see @2-verif_components) and how these components may vary under different specific situations, which is typically achieved by carefully stratifying and pooling sets of forecast-observation pairs (see #citet[@hamill_measuring_2006] for a discussion of dangers and opportunities of this sort of analysis). Forecast verification is also used, sometimes in combination with rigorous statistical testing methods, to objectively compare different forecasting systems, such as different candidates when developing a new product or entirely different products used for the same task. During day-to-day operations, as well as from a long-term perspective, forecast verification is also used to monitor the behavior of forecasting systems.

The adoption of data-driven methods to develop new products also means that metrics previously only used for post-hoc verification are now used to optimize machine learning models (see @2-machine_learning), which puts even more importance on understanding which properties of the forecasts are of more interest in order to set our optimization goals correctly. 

Finally, because the usefulness and value of a forecast increases if we are aware of its error characteristics, one could also argue that verification is itself a form of optimization. Semantics aside, it is clear that verification and optimization of forecasting systems are conceptually and practically close activities. It thus comes with no surprise that, in the meteorological value chain framework, the two activities are often carried out at similar stages and by the same individuals. They are part of the same toolbox.

In the remainder of this section, we will first learn about the most relevant aspects characterizing the performance of a forecasts, and then proceed to describe some of the most useful verification tools commonly employed during this project. In @appendix-scoringrules we present work on an open-source Python library for probabilistic forecast evaluation, `scoringrules`. Some of the work presented in @appendix-mlpp about `mlpp`, an open-source framework developed internally at MeteoSwiss to facilitate the development of ML-based post-processing models, also included the implementation of forecast verification tools.


=== Components of forecast performance <2-verif_components>
Fundamentally, all the information needed to investigate the quality of a forecast lies in the joint distribution of forecasts and observations @murphy_general_1987, which can be factorized into conditional and marginal components to gain more insights into different aspects of the forecasts. We review some of these aspects, first introduced in #citet[@murphy_what_1993], and refer readers to #citet[@wilks_statistical_2019] for a more detailed discussion.

- _Accuracy_ refers to the average correspondence between forecasts and the observation that materializes and it is meant to summarize the overall quality of a forecasting system with a single number. It is obtained by averaging scalar measures - some of which will be presented in @2-scoring_rules - computed for several forecast-observation pairs. The remaining attributes can be interpreted as components of accuracy.

- _Bias_ measures the correspondence between the average forecast and the average materialized value. 

- _Reliability_ or _calibration_ or _conditional bias_ relates to the joint distribution of observations and forecasts given specific values of the forecasts. This type of information is obtained by stratifying forecast-observation pairs based on the values of the forecasts and looking at the conditional distribution of the observations given the forecasts, $p(o | f)$.

- _Resolution_, like reliability, is related to the properties of the conditional distributions of the observations given the forecasts, but it considers them from another perspective. A forecast has high resolution if the stratification of the forecast-observation pairs based on the forecasts does indeed separate observations in distinct groups. In other words, the distribution of outcomes if event $A$ is predicted, $p(o | A)$ is different from the distribution of outcomes if event $B$ is predicted, $p(o | B)$, regardless of how accurate the prediction might be. It is a measure of the information content of a forecast.

- _Discrimination_ is the converse of resolution as it looks at the conditional distribution of the forecasts given an observed event, $p(y | E)$. It is the ability of a forecast to distinguish, or discriminate, between different situations. Like resolution, it is related to the amount of information contained in a forecast.

- _Sharpness_ is an attribute of the forecast alone, and it characterizes the tendency of a forecast to predict values that deviate from the climatology. A forecasting system that always predicts climatological values might on average have good accuracy and low bias, but it is not particularly useful as it does not contain much information. Sharpness can be viewed as a complementary and orthogonal component of performance to reliability or calibration, which is why a common goal of forecasters, particularly for the case of probabilistic forecasts, is to _maximize sharpness subject to calibration_ @gneiting_probabilistic_2007. When optimized together, they lead to accurate forecasts.

To diagnose or quantify these aspects of forecast performance, essentially two approaches exist. The first is to graphically inspect the properties of the joint distribution of forecasts and observations. Graphical visualization tools are commonly used for diagnostic verification because they provide an overview of strengths and limitations of a forecasting system, however they cannot be used as an general, objective measure of performance. To achieve this, a second approach is based on functions that compare forecasts and observations to compute scalar measures or metrics that are used to represent a forecasting system's accuracy. While most of these functions are used to provide an overall measure of performance, some of them can be decomposed into different aspects @brocker_reliability_2009.

In the following, we will provide a short and practical overview of commonly used verification tools, with a focus on probabilistic forecast evaluation. We refer readers to #citet[@jolliffe_forecast_2011] and #citet[@thorarinsdottir_chapter_2018] for a more extensive review of the topic.

=== Graphical verification tools
In the distributional view of forecast verification graphical tools play a central role, and they are typically employed to assess the calibration of forecasts. Several notions of calibration exist @gneiting_probabilistic_2007@gneiting_combining_2013, but in practice it is common to consider two main types of calibration: the conditional calibration and the weaker probabilistic calibration. Here we explain two common tools to diagnose both.



==== Reliability diagram
#figure([
  #cetz.canvas({
    import cetz.plot: *

    let ylab = par(leading: 4pt)[Observed relative \ frequency]
    ylab = align(center + horizon)[#ylab]
    
    plot(
      size: (5,5),
      // domain: (0, 1),
      x-min: 0, x-max: 1,
      y-min: 0, y-max: 1,
      x-tick-step: 0.2, 
      y-tick-step: 0.2, 
      x-label: "Forecast probability", 
      y-label: ylab, {

        // perfect line
        add(((0,0), (1,1)))
        
        // climatology line
        add(((0, 0.3), (1, 0.3)), style: (stroke: (dash: "dashed")))

        // skill area 
        add-fill-between(
          domain: (0, 1),
          ((0,0), (0, 0.17), (1, 0.6)),
          ((0.3, 0), (0.3, 1), (1,1)),
          style: (stroke: none),
        )
    })
  }) #v(0.5cm)],
  caption: [
    A conceptual representation of the reliability diagram. The blue line represents perfect calibration, where the frequency of the observations match the forecasted probabilities. Deviations from the blue line indicate a conditional bias. The dashed line represent a hypothetical climatology of the observations and corresponds to the line of zero resolution. The green area represents the region of positive skill
  ],
  placement: top,
) <2-fig-reliability>

As mentioned in @2-verif_components, reliability or calibration pertains to the correspondence between predicted probabilities and the observed frequencies of a certain event. 

In other words, we want to know whether when forecasting an event with a certain probability $p$, the event materializes indeed with that corresponding frequency $p$. When assessing the calibration of forecasts for binary outcomes, sometimes called probability forecasts, reliability diagrams are the most commonly used technique. @2-fig-reliability shows an idealized example: these diagrams plot the forecast probability of an event, on the horizontal axis, against the conditional frequency of the event occurring _given the forecast_, where the blue line corresponds to what a reliable forecaster would be @brocker_increasing_2007. Because we are looking at the probability of an event conditioned on the forecast itself, the reliability diagram is said to assess the conditional calibration. 


#figure(
  box(width: 60%)[
    #image("2-fig-corp_reliability.png")
  ],
  placement: top,
  caption: [
    An example reliability diagram computed according to the method described in #citet[@dimitriadis_stable_2021]. 
  ]
) <2-fig-corp_reliability>



In practice, reliability diagrams are usually constructed as follows:

\

1. Let $hat(p)_i in [0,1]$ be the predicted probability for instance $i$ and $y_i$ be the corresponding materialized outcome (0 or 1), where $i in {1, 2, ..., N}$ and $N$ is the total number of instances.

2. Divide the interval $[0, 1]$ into $M$ bins. The $j$-th bin covers the interval $[(j-1)/M, j/M]$ for $j in {1, 2, ..., M}$. 

3. For each instance $i$, find the bin $B_j$ such that $hat(p)_i in [(j-1)/M, j/M])$. Let $B_j$ be the set of indices of instances whose predicted probabilities fall into the $j$-th bin.

4. For each bin $j$, calculate the conditional event probability (CEP)  #math.equation(block: true, numbering: none)[
	$"CEP"(B_j) = 1/N_j sum_(i in B_j)y_i$	
] and predicted probability #math.equation(block: true, numbering: none)[
	$"pred"(B_j) = 1/N_j sum_(i in B_j)hat(p)_i$	
]

5. Create a plot with the x-axis representing the average predicted probability $"pred"(B_j)$ and the y-axis representing $"CEP"(B_j)$ for each bin $j$ and plot the points for all bins.

#v(1cm)


A common issue with this binning and counting approach is that finding the optimal way to divide the interval in step 2 is not trivial, which often leads to ad-hoc implementation decisions, a lack of reproducibility and inefficiency. Among the simplest options are to divide the interval into equally spaced bins or based on the quantiles of $p$, such as to have the same number of instances in each bin. It is also not uncommon to find manually defined values. To address this problem, #citet[@dimitriadis_stable_2021] proposed a powerful and theoretically principled approach to divide the interval in an optimal and consistent way that is also computationally efficient. In a nutshell, they use nonparametric isotonic regression and the pool-adjacent-violators algorithm to estimate conditional event probabilities (CEP), which yields a fully automated choice of bins without the need for any implementation decision. An example of a reliability diagram constructed with this approach, implemented in the `scoringrules` library described in @appendix-scoringrules, is shown in @2-fig-corp_reliability.

Reliability diagrams are the only commonly used graphical tool to consider the conditional calibration of forecasts, and they can only consider forecasts of binary outcomes. Diagnosing the conditional calibration for forecasts in the form of full probability distributions would be much more cumbersome, as it would require somehow to group many similar distributions based on their shape, for instance looking at the intervals of their first few moments. For this reason, it is more common to consider a weaker form of calibration: the _probabilistic calibration_, which brings us to the next paragraph.


==== Probability integral transform (PIT) histogram
The statistical consistency between probabilistic forecasts and the corresponding realizations can be diagnosed using the probability integral transform (PIT) @dawid_present_1984. Specifically, the PIT can be used to assess the probabilistic calibration.

In essence, the PIT is the value of the predictive cumulative distribution function (CDF) at the observation. Let $F$ denote the predictive CDF and $Y$ its corresponding observation. Then the PIT is the random variable $Z_F = F(Y)$. The PIT theorem states that if $F$ is continuous and $Y tilde F$, then $Z_F$ follows a standard uniform $cal(U)(0, 1)$ distribution. In other words, every PIT value $Z_F = F(Y)$ should represent a draw from a uniform distribution if an observation does in fact follow the predictive distribution $F$. We can therefore perform a first simple test for calibration by comparing the mean and variance of the samples PIT values to the mean and variance of $cal(U)(0,1)$ which should be 1/2 and 1/12 respectively. If a forecast has $"var"(Z_F) < 1/12$ it is overdispersive, and it is underdispersive when $"var"(Z_F) > 1/12$. It is however more common to assess the uniformity of the PIT values graphically, specifically by examining their histogram, which simply requires assigning PIT values to a discrete number of equally spaced bins in $[0,1]$. If the resulting PIT histogram appears uniform, then the forecast is probabilistically calibrated @gneiting_probabilistic_2014. A $sect$-shaped histogram indicates an overdispersive forecast and a $union$-shaped histogram an underdispersive one; a positive slope indicates underestimation and a negative slope overestimation. Importantly, it should be noted that the uniformity of the PIT histogram is a necessary but not sufficient condition for the forecaster to be ideal @hamill_interpretation_2001@gneiting_probabilistic_2007, which underscores the importance of complementing checks for conditional calibration with other characteristic of the forecast, such as probabilistic calibration or sharpness.

#TODO[]


Recently, in their work on weigthed verification tools, #citet[@allen_weighted_2022] introduced an extension of PIT histograms that allow to focus on on particular outcomes, such as considering only values above a certain threshold. With these, it is then possible to graphically assess conditional calibration. Consider a continuous random variable $Y$ with a distribution function $F$. When we examine $Y$ given that it exceeds a threshold $t$, denoted as $Y_(> t)$, it follows a new distribution $G$. This new distribution is defined as $G(x) = [F(x) - F(t)] \/ [1 - F(t)]$ for $x > t$, and $G(x) = $0 otherwise. To assess the calibration of forecasts when $Y$ exceeds the threshold $t$, we compute the conditional PIT values $G(y) = [F(y) - F(t)] \/ [1 - F(t)]$ for all $y > t$, and represent these values using a histogram. If $Y$'s conditional distribution matches the forecasted conditional distribution, the histogram should appear uniform, indicating conditional calibration. In their research, the authors discuss how forecasts that seem calibrated overall might be miscalibrated for specific outcomes, or vice-versa. These conditional PIT (cPIT) histograms were particularly useful in the evaluation of wind gust predictions in @4-results, where we wanted to focus on outcomes above a certain threshold, since very low values of wind gust are not particularly relevant in practice. In that case, we saw that forecasts appearning miscalibrated overall (when considering all outcomes) were actually much better calibrated when excluding very small values.

In the context of ensemble forecasting, an analogue tool to assess calibration is the rank histogram @hamill_interpretation_2001.


=== Scoring rules <2-scoring_rules>

In the case of deterministic forecasts, verification is generally a trivial problem. Several objective measurers of forecasts performance exist, and they all essentially come down to computing a distance, in the form of a real-valued scalar, between forecasts and observations @jolliffe_forecast_2011. Let $Omega$ be the set of all possible outcomes in a prediction problem, typically single values or a set of values in the multivariate case. Then objective measures can generally be defined as $S: Omega times Omega -> RR$, with $S(hat(y),y)$ representing the score  result for a forecast $hat(y) in Omega$ and a corresponding materialized outcome $y in Omega$. This is easily accomplished because forecasts and observations come in the same form (from the same sample space $Omega$), but in probabilistic forecasting the forecast is a predictive distribution $F in cal(F)$, with $cal(F)$ representing the set of all possible forecasts. Nevertheless, we still aim to obtain an objective measure that is a real-valued scalar, $S: cal(F) times Omega -> RR$. Functions that do that, starting from the joint space of predictive distributions and observations, are called _scoring rules_. Let $F in cal(F)$ denote a probabilistic forecast and $y in Omega$ a corresponding realized outcome. Scoring rules assign a numerical value $S(F, y)$ to each pair $(F, y)$.

When deciding on which scoring rule is more appropriate for a given problem, two important characteristics shoud be taken into consideration: propriety and locality. We briefly review these in the following, and then present some commonly used scoring rules.

==== Propriety

#citet[@murphy_note_1967] identified an important characteristic that scoring systems must have in order to uphold a fundamental principle: _a meteorologist that issues probability forecasts should express his true beliefs_. They thus introduced the theory of proper scoring rules, which was then reviewed and developed in a seminal paper by #citet[@gneiting_strictly_2007], where the authors also focused on demonstrating the importance of propriety in scientific and operational forecast evaluation. Specifically, they show how proper scoring rules encourage forecasters to be careful and honest with their assessments. In a nutshell, a scoring rule is _proper_ if the forecaster minimizes (assuming a score is negatively oriented) the expected score for an observation drawn from the distribution $G$ if they issue the probabilistic forecast $G$, rather than $F != G$. More formally,

$
EE_(Y tilde G)[S(G, Y)] <= EE_(Y tilde G)[S(F, Y)] "for all" F,G in cal(F),
$ <2-eq-proper>

where $Y$ denotes the random variable following the distribution $G$.
In other words, there is no alternative distribution $F$ that gives a better score than predicting the true distribution $G$. This seemingly obvious aspect is essential, because it means that one cannot cheat, or _hedge_, by manipulating the predictive distribution to obtain better scores, which in turn could be possible when using _improper_ scores @gneiting_strictly_2007.

Furthermore, a score is _strictly proper_ if the best score is unique, in other words if the equality in @2-eq-proper holds only if $F = G$. Strict propriety eliminates ambiguity in the forecaster's goal because the ideal forecast is clear and unique - there are not multiple distributions yielding the same expected score. Therefore, it reinforces the principle outlined above, as it does not allow the forecaster to report a distribution that has the same score while being different than their true belief.

==== Locality

The concept of locality pertains to whether the score depends only on the forecast probability assigned to the observed outcome $y$. A scoring rule $S$ is called _local_ if it can be expressed as:

$
S(P, y) = S(P(y), y)
$

where $P(y)$ is the probability assigned to the observed outcome $y$. This implies that the score depends solely on $P(y)$ and not on the probabilities assigned to other potential outcomes. Therefore, two distributions that assign the same probability to a certain outcome will have the same score, even though they might have substantially different distributions overall. Conversely, a scoring rule is _non-local_ if it depends on the entire distribution $P$, not just $P(y)$. 

Local scores are particularly valued for their simplicity and direct interpretability, as they focus on the accuracy of the probability assigned to the actual event @du_beyond_2021. Non-local scores, however, consider the overall quality of the probabilistic distribution, providing a more comprehensive assessment. Because non-local scores consider the entire probability distribution, forecasters are encouraged to assign high probability to a wider range of values around the observation $y$, which generally means that optimizing non-local scoring rules leads to smoother predictions. Furthermore, because local scores need the PDF of a distribution, they are not directly applicable in the case of ensemble forecasting.



==== Commonly used scoring rules

In this section, we introduce some of the most commonly used scoring rules in probabilistic forecasting, focusing on the prediction of continuous variables. Each of these scores has unique characteristics and applications, and understanding them is crucial for evaluating the quality of probabilistic forecasts.


The continuous ranked probability score (CRPS) is a widely used metric for evaluating probabilistic forecasts. Formally, for a given probabilistic forecast represented by the cumulative distribution function $F$ and an observed value $y$, the CRPS is defined as:

$
"CRPS"(F, y) = integral_RR [F(x) - bb(1){x >= y}]^2"d"x
$

where $bb(1)$ is the indicator function that is 1 if $x >= y$ and 0 otherwise. In practice, this integral calculates the squared difference between the forecast CDF $F(x)$ and the empirical CDF of the observation $y$, integrated over all possible values of $x$. As such, it can be interpreted as the Brier score integrated over all possible thresholds in $RR$. The CRPS is particularly attractive because it is negatively oriented and in the same units as the variable of interest, and it is a generalization of the absolute error for deterministic forecast to which it can be directly compared. Furthermore, it takes into account both sharpness and calibration, and because it evaluates the entire forecast distribution it is a non-local score @gneiting_strictly_2007.
Conveniently, analytical formulas for the computation of the CRPS exist for several parametric distributions @jordan_evaluating_2019. Nevertheless, when the true forecast CDF is not fully known, but represented by a finite ensemble, the CRPS can be estimated with some error via several estimation methods @zamo_estimation_2018, making it especially useful in the case of ensemble forecasting and distribution-free methods in general.
#citet[@allen_weighted_2022] recently reviewed and developed the concept of weighted scores, used to put more emphasis on certain outcomes of interest, and defined weighted versions of the CRPS. Ways to decompose the mean CRPS in the components outlined in @2-verif_components can be found in the literature @hersbach_decomposition_2000@arnold_decompositions_2023.
In `scoringrules`, presented in @appendix-scoringrules, we implemented many of the existing closed-forms of the CRPS and its estimators, including weighted versions.

The logarithmic score (LS) is one of the simplest and most widely used proper scoring rules. It is defined as:

$
"LS"(F, y) = "log"F(y)
$

where $F(y)$ is the probability assigned by the forecast $F$ to the observed outcome $y$. As such, the LS is a local score, as it depends solely on the probability assigned to the observed outcome. Closed form expressions for the LS exists for virtually all parametric univariate distributions, and importantly also for some multivariate distributions, such as the multivariate Gaussian distribution. In @chap-4[Chapter], Gaussian processes models heavily rely on an equivalent negatively oriented version of the LS, known as the negative log likelihood in the machine learning field, defined for the multivariate Gaussian distribution.


== Machine learning: main concepts and methods <2-machine_learning>

Broadly speaking, machine learning or ML is a field of study concerned with the development and use of statistical algorithms that can learn from data without being explicitly programmed @mitchell_machine_1997.

=== Types of machine learning

==== Supervised

Supervised learning, a subset of machine learning, involves training a model on a labeled dataset, which consists of input-output pairs \((X, Y)\). The objective is to learn a function \( f: X \rightarrow Y \) that maps inputs to outputs. In this context, \(X = \{x_1, x_2, \ldots, x_n\}\) represents the set of input features, and \(Y = \{y_1, y_2, \ldots, y_n\}\) denotes the corresponding set of target values or labels. The training process involves adjusting the parameters \(\theta\) of the model to minimize a predefined loss function \(L(Y, \hat{Y})\), where \(\hat{Y}\) represents the predicted outputs. Common loss functions include mean squared error (MSE) for regression tasks and cross-entropy loss for classification tasks.

// Mathematically, the training process aims to find the optimal parameters \(\theta^*\) by solving the optimization problem:

// \[
// \theta^* = \arg \min_\theta \sum_{i=1}^n L(y_i, f(x_i; \theta))
// \]

Here, \(f(x_i; \theta)\) denotes the model's prediction for input \(x_i\) given parameters \(\theta\). The optimization is typically performed using gradient descent or its variants, which iteratively update \(\theta\) in the direction that reduces the loss function.

Supervised learning algorithms can be categorized into two main types: regression and classification. Regression algorithms predict continuous values, while classification algorithms predict discrete labels. Examples of regression algorithms include linear regression and support vector regression, whereas examples of classification algorithms include logistic regression, support vector machines (SVM), and neural networks.

Evaluating the performance of a supervised learning model involves metrics such as accuracy, precision, recall, F1 score for classification tasks, and R-squared, mean absolute error (MAE), and root mean squared error (RMSE) for regression tasks. Proper evaluation ensures the model's generalizability to unseen data, which is critical for its deployment in real-world applications.

==== Unsupervised
Unsupervised learning, a branch of machine learning, deals with the analysis of unlabeled datasets \(X = \{x_1, x_2, \ldots, x_n\}\), where no target labels \(Y\) are provided. The objective is to identify underlying patterns, structures, or distributions within the data. Unlike supervised learning, unsupervised learning does not involve a predefined loss function or target variable. Instead, it focuses on discovering hidden features or groupings in the data.

Unsupervised learning algorithms can be broadly categorized into clustering and dimensionality reduction techniques. Clustering algorithms, such as k-means and hierarchical clustering, aim to partition the data into distinct groups or clusters. For instance, k-means clustering minimizes the within-cluster sum of squares (WCSS) to find \(k\) centroids \(\mu_1, \mu_2, \ldots, \mu_k\) and assigns each data point \(x_i\) to the nearest centroid:

\[
\arg \min_{\{\mu_j\}_{j=1}^k} \sum_{i=1}^n \min_{j \in \{1, \ldots, k\}} \|x_i - \mu_j\|^2
\]

Dimensionality reduction techniques, such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE), aim to reduce the number of features while preserving the essential structure of the data. PCA, for example, transforms the original features into a new set of orthogonal components that maximize the variance:

\[
\text{maximize} \quad \text{Var}(Z) \quad \text{subject to} \quad Z = XW, \quad W^TW = I
\]

where \(Z\) represents the transformed data and \(W\) is the matrix of eigenvectors.

Evaluating unsupervised learning models can be more challenging due to the absence of labeled data. Metrics such as silhouette score, Davies-Bouldin index for clustering, and explained variance ratio for PCA are commonly used to assess the quality of the results. Unsupervised learning is crucial for exploratory data analysis, preprocessing, and gaining insights from data without explicit labels, enabling a deeper understanding of the data's intrinsic properties.


=== Bias-variance tradeoff and overfitting
=== Gradient-based optimization and neural networks


== Feature engineering and selection <2-feature_selection>
Feature engineering and selection are critical steps in the data pre-processing pipeline that significantly impact the performance of machine learning models @cai_feature_2018. Feature engineering involves creating new features or transforming existing ones to enhance the predictive power of the model. This process requires domain knowledge and creativity to uncover hidden patterns and relationships within the data. Feature selection, on the other hand, focuses on identifying the most relevant features, thereby reducing dimensionality, improving model efficiency, and mitigating overfitting. Together, these techniques enable the construction of robust models that generalize well to unseen data, ultimately driving better decision-making and insights.

We will first look at general principles to follow when @2-general. In @2-flow and @2-static we will discuss two main categories of input features for the kind of problems that we are facing, respectively _flow-dependent_ features and _static_ features.

=== General principles <2-general>


==== Understand the domain and the data 

To be effective at feature engineering and selection it is essential to have a deep knowledge of the scientific domain of application, in our case the weather. Understanding the context, the nature of the data, and the relationships between variables allows for the creation of meaningful and relevant features, and in identifying which features are likely to be informative for the model. One of the key aspect to understand in our domain is the structure of the error and how it relates to the limitations of the weather models, which we discussed in @1-limitations.

==== Filter, simplify and reduce dimensionality

A good rule-of-thumb when determining the input features for an ML model is that the relevant predictive information should be represented by a vector that is as small as possible. Identifying and eliminating redundant or irrelevant features can make ML models more interpretable and efficient @yu_efficient_2004, and this can be done based on metrics such as correlation or mutual information @vergara_review_2014, or even simply based on expert judgment. Selecting a subset of the initial available features after analyzing their joint statistical properties is often referred to as _filter method_ approach. Alternatively, approaches based on selecting features by evaluating models with many different subsets and chosing the ones obtaining the best performance is known as _wrapper methods_ @jovic_review_2015@chandrashekar_survey_2014. 

Furthermore, dimensionality reduction techniques exist that are performed automatically via algorithms, such as principal component analysis or neural network autoencoders @jia_feature_2022. For instance, #citet[@lerch_convolutional_2022] propose using neural networks to encode complex spatial information into a low-dimensional representation that is used as input for a statistical post-processing model. 

==== Mind dataset shifts and representativeness 
In ML, dataset shifts occur when the statistical properties of data used for model training differ from those encountered during testing or operational use @quinonero-candela_dataset_2009. As the term suggests, these differences can take the form of distributional shifts, but they can also be differences in the coverage of the input space. They are important because they can severely compromise the performance of ML models, as they are forced to extrapolate on out-of-distribution predictions @cao_extrapolation_2023.

In our context, these are some common examples of dataset shifts:

- changes in the weather model can alter the distribution of the DMO prediction, used as inputs for post-processing models @demaeyer_correcting_2020;

- significant differences in weather patterns and average conditions over seasonal to yearly time scales, although they can be part of natural variability, can also result in dataset shifts. Similarly, climatological trends related to climate change may induce similar effects @beucler_climate-invariant_2024;

- in the context of global post-processing, low representativity of surface weather stations with respect to real-world conditions can affect spatial generalization capabilities.


Addressing these shifts is crucial to maintain forecast accuracy and reliability. Effective selection of predictive features may play a vital role in this context, because we can consider their stability over time and space and avoid features that are more prone to significant shifts between training and real-world data. 

An increasingly popular concept, especially in the case of ML applied to physical systems, is the one of _invariance_ @ling_machine_2016. Analogously, in the case of boundary layer meteorology, similarity theory @stull_similarity_1988 also grounds itself in the concept of invariance. The idea is essentially to identify or construct quantities that do not change (i.e. are invariant) if the frame of reference of the problem changes. For instance, #citet[@beucler_climate-invariant_2024] proposes using climate-invariant quantities whose distribution remains almost unchanged across different climates -- the simplest example being the use of relative humidity instead of specific humidity --, substantially improving the generalization capabilities of ML models.

// ==== Transform and combine raw features



=== Flow-dependent features: using the weather model state <2-flow>

As already mentioned in @1-limitations, a key idea to consider when reasoning about the errors and uncertainties of weather models is that they are both flow-dependent. That is, the statistical properties of the errors depend on the given state of the atmosphere. Therefore, if we want to develop a forecasting model that has good resolution and discriminaton (see @2-verif_components), we want the model to behave differently under different weather states. The way to accomplish this is to provide it with sufficient information about the weather state, as demonstrated by #citet[@allen_regimedependent_2019] with a theoretical study.

By design, information about the weather state is included in the input features when we include weather models' forecasts of several quantities interpolated from the model grid cells to the target location. This is usually accomplished via nearest-neighbour or linear interpolation methods. A ML model is then able to learn flow-dependently by considering patterns in the joint distribution of those inputs, which can often be indicating the presence of specific weather conditions. For instance, the combined occurrence of low pressure, high temperature and high humidity is often related to a highly unstable atmosphere. The post-processing method proposed in #citet[@hewson_low-cost_2021] exemplifies how NWP values from a single grid-cell can inform about sub-grid variability.

It is clear, however, that only considering information at a single location and time cannot fully inform the model about the weather state, particularly about meso-scale or synoptic-scale phenomena. In this case we say that the model does not have sufficient _context_. To make models more context-aware, several strategies can be used, which we will briefly review in the following sub-sections. Although we will not discuss it in detail, another dimension to the problem is that context may come from multiple ensemble members, making the context data three-dimensional ($"time" times "space" times "realization"$). Recently, #citet[@hohlein_postprocessing_2024] proposed a principled approach to include context from ensembles.



==== Patches and windows

Given that the atmospheric state in weather models lies on a spatio-temporal grid, the obvious way to include more context is to consider more grid-cells or timesteps around the prediction target, instead of interpolating to a single point. When considering the spatial dimension, it is common to adopt the term _patches_ to refer to the set of grid-cells selected around the target location. For instance, #citet[@dujardin_windtopo_2022] proposes a downscaling approach for surface wind that uses convolutional neural networks on patches centered around the target point. A problem with such an approach is that when applied to generate predictions on a dense grid it is inefficient as there's a large amount of overlap and duplicate computations, which makes it prohibitively expensive for operational use. The equivalent in the temporal dimension would be to consider a set of timesteps within a certain _window_ around the forecast time. For instance, #citet[@mlakar_ensemble_2024] take this concept to the limit and include all leadtimes of a model run as inputs to a NN-based model.

Providing models with patches and windows as inputs lets them learn useful features automatically. However, an alternative to consider is to compute before training, during a data preprocessing step, clever aggregations over those sets of grid-cell values, and include those as inputs instead.

==== Proxies and indices

A relatively easy approach to add weather context to our input space is to use single scalar values that are constructed by carefully extracting and combining specific information from the weather state, potentially across large distances. For instance, simply computing the difference of surface pressure between specific locations can be seen as a proxy for the synoptic-scale situation. In Switzerland, a notable example is the pressure gradient between the northern side and the sourthern side of the Alpine ridge -- e.g. between Zürich and Lugano --, which is highly correlated with the occurrence of Foehn @richner_understanding_2013. Alternatively, many national weather services derive so-called weather indexes or weather type classifications, which are specifically developed to diagnose the presence of recurrent weather phenomena in a given region @weusthoff_weather_2011[for instance]. Generally, expert knowledge from forecasters that are familiar with the regional situations can be particularly valuable when considering this approach to add context.


==== Encoding context in low-dimensional vectors

As we include more information from the weather state to our inputs, computational requirements increase and so does the complexity of ML models. It becomes then necessary to learn compressed, low-dimensional representations of the data. #citet[@lerch_convolutional_2022] present an example of such approach and show it brings significant improvements to their post-processing model.

Although focusing on a different, yet related problem of archiving large weather model datasets, this concept of data compression for atmospheric states has recently gained a lot of attention @mirowski_neural_2024@huang_compressing_2023@gomes_neural_2024. These techniques could potentially be used to efficiently provide flow-dependent context to post-processing models.
// Interestingly, this concept has gained attention recently, although focusing on the different but related problem of archiving 

Similarly, because DDWP models have internal low-dimensional representations of the weather state by design (more about DDWP models in @1-ddwp), we can imagine that these could be used directly by statistical post-processing models.


=== Static features:  <2-static>

When predictors are not derived from the NWP model state, but rather always known in advance for any prediction task at any location and time, we refer to them as _static_ or _auxiliary_ features. These predictors typically consist of high-resolution geospatial data that are used to account for local effects, thus addressing the model-reality gap discussed in @1-gap. Furthermore, temporal auxiliary features to account for seasonality or diurnal cycle effects are also commonly used @demaeyer_euppbench_2023[for instance].

Static features inform post-processing models in two distinct ways. Firstly, they interact with flow-dependent features and moderate how those influence a model's predictions. Given enough complexity, a ML model learns these interactions automatically, however it can be beneficial to construct features beforehand based on meteorological expertise. Secondly, they provide predictive power in climatological terms. Ideally, the first type of information should be more important when the predictions from the NWP model are sufficiently skilled. Then, a post-processing model should learn to gradually use them in the second way to converge towards climatological predictions that are specific to a certain location and time. In the following, we discuss a few examples of static features.

==== Topographic descriptors and land cover

Complex terrain is one of the most challenging aspects in weather forecasting @colman_numerical_2013, as it introduces small scale variability that is difficult to represent even by high-resolution regional models such as COSMO-1E. As such, it constitutes an important source of errors for models operated over Switzerland and more generally the Alps, which are part of our research domain. 

To address this, post-processing models make use of detailed information about the orography from a digital elevation model (DEM). During our project, we have constructed a DEM covering the swiss radar domain (red box in @1-fig-domains) by merging data from multiple sources. #citet[@swisstopo_dhm25_2023] was used for the area covering Switzerland, and a combination of #citet[@crippen_nasadem_2016] and #citet[@hengl_continental_2020] for the remainder, resulting in a product with a high resolution of 25 meters. Given a DEM, we then proceeded to extract so-called _topographic descriptors_. In a nutshell, these are features that desvcribe the local geomorphological properties of the terrain, and are obtain via basic image processing operations as well as more complex algorithms.

When deriving topographical descriptors from a given dataset, one must carefully consider the problem of dataset shift discussed in @2-general. When surface station measurements are used as targets and we aim to generalize in space (i.e. a _global_ post-processing model), the representativity of the network must be analyzed. For instance, the majority of weather stations in Switzerland are located on flatlands or on mountain tops, but rarely on slopes. As #citet[@foresti_kernel-based_2011] eloquently put it, _the network can be said homogeneous in the spatial domain but clustered in the topographic domain_.

==== Temporal encodings for seasonal and diurnal cycles

==== Interactions


// #figure(
//   cetz.canvas({
//     import cetz.draw: *

//     rect((0,0), (1,1))
//   })
// )



