
#import "../utils.typ": *

= Physics-constrained post-processing <chap-3>

This chapter has been adapted from the manuscript:
\ \
#set list(indent: 20pt)
- Zanetta, F., D. Nerini, T. Beucler, and M. A. Liniger, 2023: Physics-Constrained Deep Learning Post-processing of Temperature and Humidity. Artif. Intell. Earth Syst., 2, e220089, https://doi.org/10.1175/AIES-D-22-0089.1.
\
It addresses the topic of the physical consistency of the optimized predictions. In the manuscript, we present a methodology to ensure that post-processed forecasts of surface temperature and humidity respect fundamental laws of thermodynamics. This is achieved by including domain knowledge in the form of physical equations in the post-processing model. More generally, this manuscript shows the flexibility of differentiable machine learning in accommodating specific requirements.

#pagebreak()

#[
  
#set heading(offset: 1)


= Summary
Weather forecasting centers currently rely on statistical postprocessing methods to minimize forecast error. This improves skill but can lead to predictions that violate physical principles or disregard dependencies between variables, which can be problematic for downstream applications and for the trustworthiness of postprocessing models, especially when they are based on new machine learning approaches. Building on recent advances in physics-informed machine learning, we propose to achieve physical consistency in deep learning–based postprocessing models by integrating meteorological expertise in the form of analytic equations. Applied to the postprocessing of surface weather in Switzerland, we ﬁnd that constraining a neural network to enforce thermodynamic state equations yields physically consistent predictions of temperature and humidity without compromising performance. Our approach is especially advantageous when data are scarce, and our ﬁndings suggest that incorporating domain expertise into postprocessing models allows the optimization of weather forecast information while satisfying application-speciﬁc requirements.

= Introduction <3-introduction>
Weather forecasting centers heavily rely on statistical methods to correct and refine NWP outputs, which improves skill at low computational cost @hemri_trends_2014@vannitsem_statistical_2018. While the fundamental approach has remained the same for decades – statistically relating past NWP model outputs and other additional data, such as topographic descriptors or seasonality, to observations – the traditional divide between physical and statistical modeling is narrowing as increasingly more sophisticated models emerge to harness the growing volume of available data @vannitsem_statistical_2021.

Current research focuses particularly on machine learning techniques, with deep learning and artificial neural networks (ANNs) emerging as a modern class of post-processing methods with the potential to outperform traditional approaches in several aspects. For example, #citet[@rasp_neural_2018] found that simple feed-forward ANNs could significantly outperform traditional regression based post-processing techniques, while being less computationally demanding at inference time. The authors highlighted that ANNs could better incorporate non-linear relationships in a data-driven fashion, and were more suited to handle the increasing volumes of model and observation data thanks to their flexibility. ANNs have also been combined with other statistical techniques such as Bernstein polynomials @bremnes_ensemble_2020 for non-parametric probabilistic predictions.

Furthermore, more sophisticated ANNs such as convolutional neural networks (CNNs) have the ability to incorporate spatial and temporal data with unprecedented flexibility. #citet[@gronquist_deep_2021] used CNNs to improve forecasts of global weather. 
#citet[@hohlein_comparative_2020] and #citet[@veldkamp_statistical_2021] used CNNs for spatial downscaling of surface wind fields. A process-specific application was proposed by #citet[@chapman_improving_2019] and #citet[@chapman_probabilistic_2022], with the goal of improving the prediction of atmospheric rivers, which are filaments of intense horizontal water vapor transport @ralph_defining_2018. #citet[@dai_spatially_2021] implemented a generative adversarial network based on CNNs to produce physically realistic post-processed forecasts of cloud cover. Thus, first attempts at using DL-based approaches have shown promising improvements over traditional approaches, as they better capture non-linear dependencies and often require less feature engineering. Still, a number of challenges remain in applying ML approaches to the post-processing world @vannitsem_statistical_2021@haupt_towards_2021, and they cannot be considered a panacea for all problems. This stresses the need to include more domain expertise into data-driven approaches in a hybrid manner, which is facilitated by the availability of custom losses and architectures in standard machine learning libraries @ebert-uphoff_cira_2021.

When traditional post-processing methods are applied, the goal is to minimize the forecast error. This often leads to predictions that do not exhibit the typical spatial and temporal correlation structure that emerges from common patterns of atmospheric phenomena, or predictions that violate physical principles and dependencies between variables. However, for various applications, such as animated maps of meteorological parameters commonly disseminated to the public, or in the context of hydrological forecasting @cloke_ensemble_2009 and renewable energy @pinson_chapter_2018, it is important to provide forecast scenarios that not only have a smaller error, but also exhibit realistic spatio-temporal structures @schefzik_ensemble_2017. Furthermore, consistency across variables should be ensured in various applications. For hydrological modeling, for example, temperature, radiation, and precipitation should be consistent at all times. The issue of consistency is particularly relevant in the context of probabilistic post-processing, where sampling from marginal predictive distributions is an additional step that further breaks the spatiotemporal and inter-variable consistency. Existing approaches try to model dependencies from a statistical perspective, and not a physical one. We believe the two are complementary, closely related yet different as noted by #citet[@mohrlen_chapter_2023]. Furthermore, existing approaches were developed in the context of probabilistic forecasting, and they rely on the existence of a finite ensemble. Conversely, the methodology proposed here works in the deterministic setting, while the extension to probabilistic forecasts is left for future work. In the post-processing field, limited research is available on the issue of physical consistency, in the sense of respecting physical principles or variable dependencies based on analytic relationships. However, this question has recently gained a lot of attention in the wider ML community, and some applications in weather modeling are reviewed in #citet[@kashinath_physics-informed_2021] or #citet[@willard_integrating_2022]. In general, it has been shown that physical consistency can be pursued by applying constraints to DL models in order to prescribe specific physical processes. These constraints can take many forms. The most widely used approach is to incorporate physics via soft constraints, by defining physics-based losses in addition to common performance metrics such as mean absolute error @daw_physics-guided_2021. Another popular approach is to design custom model architectures such that the physical constraints are strictly enforced (e.g. #citet[@beucler_enforcing_2021]; #citet[@ling_reynolds_2016])

In this paper we explore ways to incorporate domain knowledge in DL-based post-processing models of temperature and humidity, and the related state variables. Specifically, we evaluate the effect of imposing constraints based on the ideal gas law and an empirical approximation of a thermodynamic state equation, and we identify benefits and disadvantages of different approaches. The goal of this paper is not to develop a highly optimized model for operational use, but rather to provide some technical guidelines and insights about incorporating meteorological expertise, in the form of analytic equations, in post-processing models of NWP. For this reason, we simplified the problem under several aspects as explained in the next section. Most importantly, we focus here on a deterministic setting, although we hope to extend this framework to probabilistic predictions in the future.

= Data and methods <3-data_methods>

#figure(
  block(inset: (right: -1cm))[#box(width: 80%, image("study-domain.png"))],
  caption: [
    The location of the 131 weather stations across Switzerland considered in our study. Stations are colored based on their elevation above sea level. We show the topography of Switzerland and its surroundings in the background. Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.
  ],
  placement: top,
) <3-fig-map>


== Datasets <3-datasets>
In this study, we make predictions at 131 weather station locations covering Switzerland, as shown in @3-fig-map. We consider forecast data from COSMO-E @klasa_evaluation_2018 for the post-processing task that spans January 2017-December 2022. COSMO-E is a limited area weather forecasting model used for the operational weather forecasts for Switzerland by MeteoSwiss. It is operated as a 21-member ensemble with a 2.2k̇m horizontal resolution. It runs two cycles per day (0000 and 1200 UTC) with a time horizon of five days. We interpolate the nearest model grid-cell to the selected station locations, using the $k$-d tree method. We retain model runs initialized at 0000 UTC and consider leadtimes between +3 and +24 hours with hourly steps. We use observational data from the SwissMetNet network to define our "ground truth". The predictors and predictands used in this study are summarized in @3-tab-mapping. They represent instantaneous measurements with hourly granularity. We note that while using both dew point temperature and dew point deficit is redundant, this is necessary to guarantee a fair comparison between the different model architectures in @3-results. In contrast to prior post-processing studies, we train a single model for all leadtimes and use the leadtime as a predictor, thereby increasing the variance in the training set. This choice was motivated by the ease of implementation, allowing us to focus on the main aspect of the proposed methodology, namely the physical constraints.

== Cross-validation and random seed <3-cross_val_random>
In order to consider the variability due to the data used during optimization, each model configuration was trained on multiple cross-validation splits. When dealing with time series, it is important to design the cross-validation strategy in a way that ensures independence between sets. At the same time, in our setup it is desirable that samples of different sets are roughly equally distributed throughout the year. To do so, we opted for a simple four-folds cross-validation with a holdout set for testing. Specifically, of our five years of data we used 20% for testing, 60% for training and 20% for validation, and we removed values between sets in a gap of five days in order to ensure independence. A total of four years was considered for training and validation, meaning each of the four cross-validation folds considered a different year in the dataset. 
#TODO[How to include SM content?]
The dataset partitioning is shown in the Supplementary Material’s (SM) Fig. SM1. For each cross-validation split, we applied a standard normalization to the inputs of each set based on the training set’s mean and standard deviation. Moreover, to account for the stochasticity of neural networks optimization (due to weights initialization and gradient descent), each model configuration was trained with 3 different random seeds. In total, for any given approach, 12 trials were conducted with the trained models and subsequently all evaluated on the same holdout test dataset.



#figure(
  table(
  columns: 3,
    align: (col, row) => (left, left, left,).at(col),
    row-gutter: -6pt,
    inset: 6pt,
    stroke: none,
    table.hline(start: 0, stroke: 1.5pt),
    table.header(
      [Predictors and Predictands],
      [Units],
      [Symbol],
    ),
    [],
    [],
    [],
    table.hline(start: 0),
    table.cell(
      colspan: 3, align: center, text(style: "italic")[Predictors]),
    [Air temperature ensemble average],          [°C],        [$T$],
    [Dew point temperature ensemble average],    [°C],        [$T_d$],
    [Dew point deficit ensemble average],        [°C],        [$T_(upright(d e f))$],
    [Relative humidity ensemble average],        [%],         [$R H$],
    [Surface air pressure ensemble average],     [hPa],       [$P$],
    [Water vapor mixing ratio ensemble average], [g kg^-1],    [$r$],
    [Forecast leadtime],                         [h],         [],
    [Sine component of day of the year],         [],          [],
    [Cosine component of day of the year],[],[],
    [Sine component of hour of the day], [], [],
    [Cosine component of hour of the day], [], [],
    [],[],[],
    table.cell(
      colspan: 3, align: center, text(style: "italic")[Predictands]),
    
      [2m Air temperature],
    [°C],
    [$T$],
    [2 m Dew point temperature],
    [°C],
    [$T_d$],
    [2 m Surface air pressure],
    [hPa],
    [$P$],
    [2 m Relative humidity],
    [%],
    [$R H$],
    [2 m Water vapor mixing ratio],
    [g kg$""^(- 1)$],
    [$r$],
      ),
    caption: figure.caption(
      position: top, 
      [List of predictors and predictands considered in this study.]
    ),
    placement: top,
) <3-tab-mapping>


== Multi-task neural networks for post-processing <3-multi_task>
To keep our DL framework general, the basic building block used for all models in this study is a fully connected neural network (FCN, see @3-fig-methods) that takes as inputs a vector containing the predictors in @3-tab-mapping and a vector with station IDs. Station IDs are mapped into a n-dimensional, real-valued vector $bold("z") in bb(R)^n$ via an embedding layer and concatenated with the predictors. This approach, here referred to as \"unconstrained setting\", is the same proposed by #citet[@rasp_neural_2018] and may be regarded as state-of-the-art for the case of local post-processing of surface variables @schulz_machine_2022. It is convenient because it allows for training a single model to do local post-processing at many stations, instead of training one model for each station, making its operational implementation easier. We note that our chosen embedding is embedding is $bb(R)^6$ which is more dimensions than the $bb(R)^2$ found in #citet[@rasp_neural_2018]: this is likely due to the fact that we target five variables simultaneously and in part due to the complex topography of the Alps (compared to German stations). The forecasts are deterministic, and we train a model to predict multiple target variables simultaneously. As such, we are dealing with a case of multi-task learning @crawshaw_multi-task_2020, where the individual errors of each task contribute to the objective loss function. #citet[@kendall_multi-task_2018] observed that the relative weighting of each task’s loss had a strong influence on the overall performance of such models. This is fairly intuitive in our application because tasks have have different scales (due to different units) and uncertainties. The authors proposed a weighting scheme based on the homoscedastic uncertainty of each task, where the weights are learned during optimization. To the best of our knowledge, this approach for multi-task learning is new in the post-processing field. If jointly post-processing multiple meteorological parameters simultaneously becomes more common, it will be important to design optimal weighting schemes. We use the m squared error (MSE) for the loss function $cal(L)_k$ of each task $k$. For a predicted value $hat(y)_(k , i)$ in physical units (we will use the hat notation for predicted values throughout the rest of the paper) and an observed value $y_(k , i)$, the task loss $cal(L)_k$ is hence defined as:

$ cal(L)_k =^"def" sum_(i = 1)^(N_"samples") lr((y_(k , i) - hat(y)_(k , i)))^2 / N_"samples" , $ 

where $N_(upright(s a m p l e s))$ is the number of samples. For $p$ tasks, we define the combined loss $cal(L)$ as:

$ cal(L) =^(upright(d e f)) sum_(k = 1)^p lr([1 / 2 cal(L)_k / sigma_k^2 + log sigma_k]) . $ <3-eq-wloss>

Each task’s loss $cal(L)_k$ is scaled by the homoscedastic uncertainty represented by $sigma_k^2$, and a regularising term $log sigma_k$ is added to prevent degenerating towards a zero-weighted loss. In practice, for improved numerical stability, we learn the log variance $eta_k =^(upright(d e f)) log sigma_k^2$ for each task $k$ and @3-eq-wloss becomes:

$ cal(L) = sum_(k = 1)^p frac(cal(L)_k exp lr((- eta_k)) + eta_k, 2) , $

where the division by 2, during the actual implementation, can be ignored for optimization purposes as it does not influence the minimization objective. To avoid having to learn large biases, we initialize the bias vector in the model’s output layer using the training set-averaged output vector, which facilitates optimization. The optimization is insensitive to the initial values of this bias vector as long as it has the same order of magnitude as the mean output.

== Enforcing analytic constraints in neural networks <enforcing-analytic-constraints-in-neural-networks>
The methodology used here follows #citet[@beucler_enforcing_2021] and was first applied to neural networks emulating subgrid-scale parametrization for climate modeling. Conservation laws are enforced during optimization via constraints in the architecture or the loss function. In this study we aim to enforce dependencies between variables using the ideal gas law and an an approximate integral of the Clausius-Clapeyron equation used operationally at MeteoSwiss. Specifically, we post-process air temperature ($T$, °C), dew point temperature ($T_d$, °C), surface air pressure ($P$, hPa), relative humidity ($R H$, %) and water vapor mixing ratio ($r$, g kg$""^(- 1)$). We then aim to enforce the following constraints:

$ R H = f lr((T , T_d)) = exp lr((frac(a dot.op T_d, b + T_d) - frac(a dot.op T, b + T))) ,\
r = g lr((P , T_d)) = frac(
  622.0 dot.op c dot.op exp lr((frac(a dot.op T_d, b + T_d))),
  P - c dot.op exp lr((frac(a dot.op T_d, b + T_d))),

) , $ <3-eq-constraints>

where $a$, $b$ and $c$ are empirical coefficients, as explained below. The system of interest includes five variables and two constraints, which leaves us with three degrees of freedom. The constraints functions $f$ and $g$ are derived from the following equations:

$ e = c dot.op exp lr((frac(a dot.op T_d, b + T_d))) quad upright(" and ") quad e_s = c dot.op exp lr((frac(a dot.op T, b + T))) , $ <3-eq-tetens>

$ R H = e / e_s dot.op 100 , $ <3-eq-RH>

$ r = 1000 dot.op frac(0.622 dot.op e, p - e) . $ <3-eq-r>

We note from @3-eq-tetens that the parameters of interest are linked by two additional physical quantities: the water vapor pressure ($e$, hPa) and the saturation water vapor pressure ($e_s$, hPa). @3-eq-tetens is structurally identical to the August–Roche–Magnus equation, an approximate integral of the Clausius-Clapeyron relation accurate for standard weather conditions. We use $a = 17.368$, $b = 238.83$ and $c = 6.107$ hPa for $T gt.eq 0$; $a = 17.856$, $b = 245.52$, $c = 6.108$ hPa otherwise. We made this choice for the coefficients to ensure consistency with MeteoSwiss’ internal processing of meteorological variables, but other values can be found in the literature @lawrence_relationship_2005. @3-eq-r is a formula for water vapor mixing ratio derived from the ideal gas law for dry air and water vapor, and can be found in many common textbooks @emanuel_atmospheric_1994. We multiply by 1000 to express $r$ in g kg$""^(- 1)$ rather than g g$""^(- 1)$.

#figure(
  box(width: auto, image("methods-diagram.png")),
  placement: top,
  caption: [Summary of the models used in this study. (a) The basic building block of all models, a fully connected network preceded by an embedding layer. (b) The unconstrained setting used as a baseline, where all target variables are predicted directly. (c) The architecture-constrained setting, including a physical constraints layer that takes a subset of the target variables as inputs and returns the complete prediction. (d) The loss-constrained neural network, in which physical consistency is enforced by adding a physics-based penalty P to the conventional loss L. (e) An offline-constrained neural network, where constraints are only applied after training using the constraints layer.]
) <3-fig-methods>
We proceed to implement and compare two approaches how to enforce physical constrains in our networks:

=== Architecture-constrained setting: <architecture-constrained-setting>
the constraints are enforced by using a layer that uses @3-eq-constraints to derive $R H$ and $r$ from $T$, $T_"def"$ and $P$. $T_"def")$ is the dew point deficit, to which we apply a ReLU activation function to ensure positivity before computing dew point temperature as $T_d =^"def" T - T_"def"$. This additional step allows us to enforce that $T gt.eq T_d$ and $R H in lr([0 , 100])$, two desirable properties in this case. We show the constrained architecture in #ref(<3-fig-methods>)c. The trainable part of the model has the same number of layers and units as the unconstrained architecture, but directly predicts only a subset of variables. In our case, which has 5 outputs and 2 constraints, we directly predict 3 variables and derive the last 2 via a custom-defined layer that encodes @3-eq-constraints. An important point is that the choice of which variables are computed first and which are derived analytically is arbitrary. Given $n = 5$ variables and $q = 2$ constraints, the total number of possibilities in our case is $frac(n !, q ! lr((n - q)) !) = 10$. Nevertheless, there are differences in the actual implementation that point in favor of some configurations. For example, one may want the analytic constraints arranged in a way such that numerical stability is not a concern, e.g., avoiding division by zero or asymptotes of logarithmic functions.

=== Loss-constrained setting <loss-constrained-setting>
the constraints are enforced by using an additional physics-based loss term that includes a penalty term $cal(P)$ based on residuals from our set of analytic equations. As for the architecture-constrained approach, the variables that we choose to calculate the residuals are arbitrary. Based on @3-eq-constraints, we define the following constraints:

$ cal(P)_(R H) =^(upright(d e f)) hat(R H) - f lr((hat(T) , hat(T_d))) = 0 ,\
cal(P)_r =^(upright(d e f)) hat(r) - g lr((hat(P) , hat(T_d))) = 0 , $

where physical violations result in non-zero residuals. Using the L2-norm for consistency with our MSE loss, we formulate the penalty term $cal(P)$ used in the loss function as:

$ cal(P) =^(upright(d e f)) lr((cal(P)_(R H)))^2 / sigma_(R H)^2 + lr((cal(P)_r))^2 / sigma_r^2 , $

where we square the residuals $cal(P)_(R H)$ and $cal(P)_r$ to penalize larger violations more, and then scale by the variance of the observed values $sigma_(R H)^2$ and $sigma_r^2$ in order to normalize the contribution of the two terms. Finally, the physical penalty term is added to the conventional loss and our training objective becomes minimizing the physically-constrained loss function $cal(L)_(cal(P))$:

$ cal(L)_(cal(P)) =^(upright(d e f)) lr((1 - alpha)) cal(L) + alpha cal(P) , $

where $alpha in lr([0 , 1])$ is a hyperparameter used to scale the contribution of the physical penalty term. Note that in contrast with the #emph[hard] constraints in the architecture, with the #emph[soft] constraints in the loss, we have no guarantee that $cal(P) = 0$ because stochastic gradient descent does not generally lead to a zero loss.

=== Offline constrained setting
Finally, to assess whether enforcing constraints #emph[during] optimization is advantageous, we additionally introduce an offline-constrained setting (see #ref(<3-fig-methods>)e in which constraints are enforced _after_ training. Specifically, we train a model to minimize the MSE of $T$, $T_d$ (derived from $T$ and $T_"def"$) and $P$. $R H$ and $r$ are then calculated after training so as to exactly enforce our physical constraints.

== Libraries, hyperparameters and training <libraries-hyperparameters-and-training>


#figure(
    align(center + horizon)[#table(
        columns: 3,
        align: (col, row) => (left, left, left,).at(col),
        inset: 6pt,
        [#strong[Parameter]],
        [#strong[Value]],
        [#strong[Search]],
        [Learning rate],
        [0.0007],
        [\[0.01-0.001\] $tilde.op $ loguniform],
        [Batch size],
        [512],
        [{256, 512, 1024, 2048} $tilde.op $ choice],
        [Units in layer 1],
        [256],
        [{32, 64, 128, 256} $tilde.op $ choice],
        [Units in layer 2],
        [256],
        [{32, 64, 128, 256} $tilde.op $ choice],
        [Embedding],
        [6],
        [{2, 3, 4, 5, 6} $tilde.op $ choice],
        [Patience],
        [5 epochs],
        [],
        [$alpha$ (loss-constrained)],
        [0.995],
        [],
      )],
    placement: top,
    caption: [Hyperparameters used to train the models, along with their optimal value and the search space used for tuning the unconstrained model, represented as the range or set of possible values followed by the sampling method. After selecting the five best configurations automatically, we chose the one with the lowest number of trainable parameters for parsimony.],
) <3-tab-hpsearch>

We use the PyTorch deep learning library @paszke_pytorch_2019 to implement our models, the Ray Tune library @liaw_tune_2018 for hyperparameter tuning, and Snakemake @molder_sustainable_2021 to manage our workflow #footnote[code at #link("www.github.com/frazane/pcpp-workflow")]. For training we use the Adam optimizer @kingma_adam_2017 with the exponential decay rates set to $beta_1 = 0.99$ and $beta_2 = 0.999$ for the first and second moments, implement an early-stopping rule based on a validation loss to avoid overfitting. We use the aggregated normalized mean absolute error (NMAE) aggregated over all 5 outputs as our validation loss:

$ upright(N M A E) =^(upright(d e f)) 1 / 5 sum_(k = 1)^5 sum_(i = 1)^(N_(upright(s a m p l e s))) lr(|y_(i , k) - hat(y_(i , k))|) / sigma_k , $

and halt training after 5 epochs in the absence of improvements in the validation loss. This metric was chosen for the validation and early stopping because it proved to be more robust to sudden model changes during training, compared to the training loss. We save the model state with the lowest validation loss of all training epochs. The hyperparameters used to produce the main results of this study are shown in @3-tab-hpsearch. We chose them after running a hyperparameter tuning algorithm for the unconstrained model that considered the aggregated loss of all cross-validation splits. The best performing hyperparameters configuration was then applied to all models. The loss-constrained model also required a hyperparameter to scale the influence of the physics-based penalty term. After testing different values, $alpha$ was set to 0.995. We discuss this choice in the next section.


= Results and discussion <3-results>

In this section, we present the results of our models when evaluated on unseen data. We will first compare the performance and physical consistency of different architectures in @3-predictive_performance before discussing data efficiency in @3-data_efficiency and generalization ability @3-generalization.

== Predictive performance and physical consistency <3-predictive_performance>

We use two metrics to evaluate the overall performance of our models: the mean absolute error (MAE) and the Mean Squared Skill Score (MSSS) calculated with respect to the raw NWP forecast, defined as:

$ upright(M S S S) = 1 - upright(M S E_(P P)) / upright(M S E_(N W P)) , $

where $upright(M S E_(P P))$ and $upright(M S E_(N W P))$ represent the MSE for our postprocessed forecasts and the NWP forecast respectively. The MSSS presented here was first computed on each station individually and then averaged. Values closer to 1 are better, and negative values indicate a decrease in performance. The overall results are presented in @3-tab-results and #ref(<3-fig-results>)a for each variable. For both MAE and MSSS these results show that the performance is comparable for all NN architectures. There are small differences in performance of each setting on the different tasks, which we hypothesize are related to the influence of the constraints coupled with the multi-task loss weighting. Overall, we observe slightly better results for the loss and architecture constrained approaches.

@3-tab-tests displays Diebold-Mariano predictive performance test results for MAE, applied individually to each station and lead time following the implementation of #citet[@schulz_machine_2022]. The reported values show the percentage of tests where an approach significantly outperformed another. Overall, these results align with @3-tab-results, without a clear winner, but the offline constrained approach appears generally worse, except for pressure. Our models achieved MAEs of approximately 1.35 °C, whereas an altitude-corrected NWP forecast, often used as a reference, gave 1.63 °C (using a fixed lapse rate of 6 °C km$""^(- 1)$).

We note that the MSSS values are surprisingly high, especially for pressure. These high MSSS values are due to the large errors in the NWP model. For variables that are strongly tied to elevation, such as pressure and temperature, the differences in the NWP model elevation and the true station elevation result in consistently large biases. These elevation differences can be larger than 100 meters. Taking the example of pressure, the mean bias at certain stations is almost 90 hPa, which is reduced to almost 0 hPa by the post-processing models, explaining the high MSSS values.



  
#figure(
    align(
      center + horizon,
    )[#set text(size: 11pt)
      #let dh(body) = {
        table.cell(align: center, colspan: 2, inset: 7pt)[#body]
      }
      #table(
        columns: 11,
        stroke: none,
        gutter: 10pt,
        align: (col, row) => (left, left, left, left, left, left, left, left, left, left, left,).at(col),
        inset: (x: -1pt, y: -1pt),
        [], dh($T$), dh($T_d$), dh($P$), dh($R H$), dh($r$),
        table.hline(start: 1, end: 3, position: bottom, y: 0),
        table.hline(start: 3, end: 5, position: bottom, y: 0),
        table.hline(start: 5, end: 7, position: bottom, y: 0),
        table.hline(start: 7, end: 9, position: bottom, y: 0),
        table.hline(start: 9, end: 11, position: bottom, y: 0),
        [], [MAE], [MSSS], [MAE],[MSSS], [MAE],[MSSS], [MAE], [MSSS], [MAE],[MSSS],
        table.hline(start: 0),
        table.cell(colspan: 11)[],
        [Unconstrained],            [1.355], [0.351], [1.514], [0.318], [0.529], [0.618], [8.737], [0.290], [0.550], [0.252],
        [Architecture constrained], [*1.340*], [*0.357*], [1.517], [0.305], [*0.527*], [*0.624*], [*8.677*], [0.286], [0.547], [0.247],
        [Loss constrained],         [1.343], [0.356], [*1.501*], [*0.326*], [0.530], [0.623], [8.722], [*0.297*], [*0.539*], [*0.274*],
        [Offline constrained],      [1.356], [0.341], [1.530], [0.295], [0.533], [0.607], [8.911], [0.261], [0.551], [0.238],
        [], [], [], [], [], [], [], [], [], [], [],
      )],
    placement: top,
    caption: [Two performance metrics: The mean absolute error (MAE; lower values are better) and mean-square skill score (MSSS; higher values are better) for each considered NN architecture (rows) and target variable (columns), averaged over the test set. Boldface type indicates the best performance for each metric and variable.]
) <3-tab-results>

#figure(
  box(width: 100%, image("default_test_combined_mixed.png")),
  placement: top,
  caption: [
    (a) MAE for each target variable and approach, where the boxplot distribution represents the nine trials using different crossvalidation splits and random seeds. Note that the ranges of these distributions are relatively small in comparison with the absolute values of the error metric. (b) Scatterplot representing the distribution of physical violations PRH in RH units, as a function of RH, using all samples of all trials, where points are color coded by density. Physical violations are deviations from the zero line.
  ]
) <3-fig-results>


#figure(
  [
    #set text(size: 8pt)
    #table(
      columns: 6,
      align: (col, row) => (left, center, center, center, center, center, center).at(col),
      stroke: none,
      // row-gutter: -5pt,
      table.hline(),
      table.header([], [Unconstrained], [Architecture constrained], [Loss constrained], [Offline constrained], [Winning avg]),
      table.hline(),
      
      table.cell(colspan: 6, align: center)[_Air temperature_],
      [Unconstrained],[],[6.72],[4.91],[15.67],[9.10],
      [Architecture constrained],[32.42],[],[16.61],[34.13],[27.72],
      [Loss constrained],[26.75],[12.69],[],[33.33],[24.26],
      [Offline constrained],[14.32],[6.83],[8.98],[],[10.04],
      [Losing avg],[24.50],[8.75],[10.17],[27.71],[],
      
      table.cell(colspan: 6)[],
      table.cell(colspan: 6, align: center)[_Dewpoint temperature_],
      [Unconstrained],[],[12.87],[6.47],[22.86],[14.07],
      [Architecture constrained],[6.18],[],[3.82],[17.92],[9.31],
      [Loss constrained],[26.32],[33.62],[],[49.07],[36.34],
      [Offline constrained],[2.29],[1.85],[1.27],[],[1.81],
      [Losing avg],[11.60],[16.12],[3.85],[29.95],[],
      
      table.cell(colspan: 6)[],
      table.cell(colspan: 6, align: center)[_Surface air pressure_],
      [Unconstrained],[],[28.50],[30.61],[32.13],[30.41],
      [Architecture constrained],[32.82],[],[36.28],[38.75],[35.95],
      [Loss constrained],[32.82],[28.10],[],[36.17],[32.36],
      [Offline constrained],[22.90],[22.43],[28.28],[],[24.54],
      [Losing avg],[29.52],[26.34],[31.72],[35.68],[],
      
      table.cell(colspan: 6)[],
      table.cell(colspan: 6, align: center)[_Relative humidity_],
      [Unconstrained],[],[9.63],[11.89],[38.10],[19.87],
      [Architecture constrained],[21.12],[],[19.81],[57.00],[32.64],
      [Loss constrained],[17.59],[14.07],[],[45.91],[25.86],
      [Offline constrained],[8.65],[1.82],[5.74],[],[5.40],
      [Losing avg],[15.79],[8.51],[12.48],[47.00],[],
      
      table.cell(colspan: 6)[],
      table.cell(colspan: 6, align: center)[_Water vapor mixing ratio_],
      [Unconstrained],[],[7.74],[1.45],[10.58],[6.59],
      [Architecture constrained],[20.39],[],[4.98],[19.12],[14.83],
      [Loss constrained],[57.76],[32.82],[],[45.69],[45.43],
      [Offline constrained],[6.47],[3.96],[1.09],[],[3.84],
      [Losing avg],[28.21],[14.84],[2.51],[25.13],[],
    )
  ],
  placement: top,
  caption: [For each target variable, the percentage of tests for which an approach (rows) produces significantly better forecasts than another (columns), according to Diebold–Mariano statistical tests performed with the MAE. Also reported is the average percentage of “wins” or “losses” for each approach. Tests are applied to each station and lead time individually.]
) <3-tab-tests>


#ref(<3-fig-results>)b depicts the physical consistency of the predictions. On the vertical axis is $cal(P)_(R H)$, which is the difference between the predicted $R H$ and its physically-consistent counterpart derived from the constraint function $f lr((T , T_d))$, while we show predicted values on the horizontal axis. Deviations from zero are therefore considered physical violations, as well as values that are not between 0 and 100. Compared to the unconstrained approach, we observe a noticeable decrease of violations using the loss-constrained approach, although large violations still occur at the ends of the RH distribution, where values larger than 100% still occur. The architecture-constrained models bring physical violations to zero to within machine precision. These results are consistent  #citet[@beucler_enforcing_2021]. As a side remark, we note that in the unconstrained approach larger violations tend to occur more at the tails of the distribution, which could indicate that it is more difficult to converge to a physically consistent solution if the samples are scarce.

#figure(
  box(width: 70%, image("p_mae_vs_alpha.png")),
  caption: [
    Effect of the hyperparameter $alpha$ on both the overall physical violation $cal(P)$ (in red) and the NMAE (in blue) for the test dataset.
  ],
  placement: top,
)
<3-fig-alpha>

In order to choose an optimal value for the $alpha$ hyperparameter, we tested several values and compared both the NMAE and the physical consistency of the predictions. The results are shown in @3-fig-alpha for the following values: $alpha in { 0 . , 0.2 , 0.5 , 0.8 , 0.9 , 0.95 , 0.99 , 0.995 , 0.999 , 0.9999 , 0.99999 }$. We observe that the trade-off between physical constraints and performance is non-linear: up to $alpha = 0.995$, there is little to no drawback in terms of performance. In contrast, for higher values of $alpha$, the NMAE starts to increase. We note two things: first, the choice of this hyperparameter $alpha$ can be chosen based on how much one wishes to prioritize physical consistency over error reduction. Second, one should be aware that the choice of the learning rate has a significant influence on $alpha$’s impact (and vice-versa) and thus on this trade-off, although this was not further investigated in this study. We limit ourselves to observe that from a practical standpoint, this relationship is inconvenient as it makes model selection harder, and we consider it to be a drawback of the loss-constrained approach.

== Robustness to data scarcity <3-data_efficiency>

#figure(
  box(width: 100%, image("data_efficiency_test_combined_mixed.png")),
  placement: top,
  caption: [(a) NMAE for each reduction and approach, where the box-plot distribution represents the nine trials using different cross-validation splits and random seeds. As the size of the training dataset reduces, constrained models perform relatively better. (b) Box-plot showing the physical penalty term P’s distribution as a function of training data size for the unconstrained and loss-constrained settings. The architecture- and offline-constrained approaches have zero penalty by construction.]
) <3-fig-data_efficiency>

Among the potential advantages of constraining neural networks using physical knowledge are improved robustness to data scarcity. The rationale is that because we reduce the hypothesis space of the model to a subset of physically-consistent solutions, we could expect the physics-constrained models to learn with fewer training samples, or to require fewer parameters. We have therefore designed an experiment in which we re-trained all models (with fewer parameters in order to reduce the chance of overfitting) on increasingly reduced training datasets, namely 20%, 5% and 1% of the full dataset. The reduction is applied by station, in order to ensure that all stations are still equally represented in the dataset. For instance, with the 1% reduction we trained with roughly 200 samples for each station. The results, shown in #ref(<3-fig-data_efficiency>)a, seem to indicate a relatively smaller decrease in performance for the architecture-constrained approach when the data is scarce, although this difference is rather small. Importantly, the added value of enforcing physical consistency in data scarce situations is emphasized by #ref(<3-fig-data_efficiency>)b. For the unconstrained model in particular, the physical inconsistencies increase as we reduce the number of training samples. Conversely, for the architecture constrained approach, physical inconsistencies are always zero by construction.

== Generalization ability <3-generalization>

A common finding of physics-informed ML is that physically-constraining models could help them generalize to unseen conditions @willard_integrating_2022. In order to test the ability of our models to generalize to unseen weather situations, we design an experiment in which models are trained on a dataset that excludes the warm season (JJA), and then tested on the warm season only. This choice was motivated by the increasing relevance of record-shattering heat extremes in a warming climate. In such situations, the robustness of post-processing models is put to test as they have to process and predict values never seen during training. We present our results in @3-fig-generalization, and observe that physical constraints do not seem to impact the generalization capabilities of the model to unseen temperature extremes. This result, consistent with #citet[@beucler_towards_2020], suggests that the constraints of @3-eq-constraints are insufficient to guarantee generalization capability for our mapping.

#figure(
  box(width: 80%, image("NMAE_vs_quantile.png")),
  caption: [
    NMAE for the test dataset containing samples from JJA, conditioned on different quantiles of the univariate temperature distribution. As expected, the error increases as temperature are more extremes, but the relative performance of the considered architectures does not change significantly.
  ],
  placement: top,
) <3-fig-generalization>

= Conclusion and outlook <conclusion-and-outlook>
In this study, we have adapted a physically-constrained, deep learning framework to postprocess weather forecasts, which is new to our knowledge. More generally, we demonstrated simple ways to integrate scientific expertise, in the form of analytic equations, into a DL-based post-processing model. Compared to unconstrained or loss-constrained models, architecture-constrained models enforce physical consistency to within machine precision without degrading the performance of the considered variables. Additionally, the architecture-constrained models were easier to implement in our case and therefore recommend them over the loss-constrained counterpart. We have also shown that physical constraints yield better predictions when data are scarce because of the increased value of physical consistency. However, we did not observe a significant advantage in terms of generalization capabilities. To interpret these results, it is useful to distinguish the data efficiency and the generalization experiment by their underlying challenge, that is, interpolation and extrapolation, respectively. Physically constraining outputs can help the model better interpolate data, but cannot mitigate the well-known limitations of neural networks when it comes to out-of-distribution inputs (extrapolation).

We believe that a significant value of the proposed methodology lies in its simplicity: any kind of equation, as long as it is differentiable, can be included in DL-based post-processing models. Importantly, this extends beyond the context of meteorology and physics-based constraints, as we could easily imagine a similar methodology used to satisfy a diverse set of constraints defined by the end users. For future research on this topic we foresee (i) an extension to probabilistic forecasting, e.g. by adopting a generative approach for the creation of physically consistent ensembles; and (ii) an extension to a global post-processing setup, where the model generalizes in space. Finally, an open question is whether physical constraints have a stronger effect in more challenging tasks, e.g. with higher-dimensional mappings or more marked non-linearities.

=== Acknowledgements
We thank the members of APPP team at MeteoSwiss for helpful comments and feedback that significantly helped the project. FZ is supported by MeteoSwiss and ETH Zürich, DN and ML are supported by MeteoSwiss, while TB is supported by the Canton of Vaud in Switzerland. We also thank the Swiss National Supercomputing Centre (CSCS) for computing infrastructure.

=== Data and code availability 
The project’s GitHub repository is accessible at #link("https://github.com/frazane/pcpp-workflow"). The raw data used to train the models is free for research and education purposes under request to MeteoSwiss.

]