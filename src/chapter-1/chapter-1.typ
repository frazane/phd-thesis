#import "@preview/cetz:0.2.2"
#import "../utils.typ": *

= Introduction <chap1>
This introduction aims to motivate this thesis and provide an overview of its primary topics. In @1-limitations, we will discuss the current limitations of weather prediction models, briefly explaining how they work and identifying the main sources of uncertainty and errors. Next, in @1-usefulness, we will focus on explaining why current models alone are restrictive in relation to our requirements and highlight the significance of statistically optimized predictions. Moving on to @1-postprocessing, we will provide some background into the field of weather forecasts post-processing, where we will look at the key concepts and attempt to organize the large variety of existing methods. Finally, in @1-scope, we will define the scope and objectives of this thesis and outline its content in @1-outline.

#pagebreak()

== Weather prediction models and their limitations <1-limitations>

Numerical weather prediction (NWP) models are the backbone of modern meteorology, with profound implications for our scientific understanding of the discipline and its practical utility. These models represent the state of Earth's atmosphere as a collection of spatially distributed variables -- wind, temperature, pressure, and humidity -- and apply our best understanding of the underlying physical laws to simulate the evolution of the state over time, starting from initial conditions known from meteorological observations. 

Our knowledge about the physical laws governing the atmosphere is not perfect. This, in conjunction with limitations of technical nature, makes it necessary for us to simplify the problem and reduce the complexity of the simulation.
Some prior simplifications are made when establishing the set of equations that govern the dynamics of the atmosphere. These are typically physical equations of motion, mass conservation, energy conservation, water vapor conservation, and the equation of state. Given some specific assumptions, these equations are simplified by ignoring some terms or interactions between terms, usually on the base that their contribution to the overall system is negligeable compared to magnitudes of interest.

Because the system of equations lacks an analytical solution, another approximation is introduced by the numerical method used to solve the system. This involves discretization in space and time, meaning that the atmospheric state is represented by a finite number of points structured in a three-dimensional grid, and that the state evolves with fixed time increments. As a consequence, some processes -- exchanges of momentum, heat, and water between the atmosphere, land, and space -- occur at scales smaller than the scale resolved by the discretized equations, or _sub-grid scales_. These must be then _parameterized_ -- approximated via a combination of theoretical and empirical relationships that describe the average effect of these exchanges at the grid scale @stensrud_parameterization_2007.

One of NWP's key elements is the ability to evolve the atmospheric state over time; the other is the generation of the initial state. 
This step involves collecting observations from various sources -- both in-situ and remote sensed -- in a temporal window prior to the starting time. Then, these data are used to compute the model's initial conditions with a procedure known as data assimilation. 
For a more detailed introduction to NWP models, we refer to #cite(<coiffier_fundamentals_2011>, form: "prose"), and to #cite(<olafsson_uncertainties_2021>, form: "prose") for a focus on model uncertainties.

In recent decades, NWP models have steadily improved in forecast skill, thanks to advances in several key aspects @bauer_quiet_2015. Computations are performed on higher resolution grids, physical parameterizations of unresolved processes are increasingly detailed and precise, the development of complex data assimilation systems and the increased quantity and quality of observations improved model initialization, and the adoption of ensemble forecasts now allows to estimate the flow-dependent uncertainty. 
Being able to perform computations on higher-resolution grids has been a particularly important factor, made possible by the fast technological advances in computing. As discussed in #cite(<neumann_assessing_2019>, form: "prose"), only improving model resolution enables significantly better simulation of certain small-scale but relevant phenomena. 
The transition from deterministic to probabilistic, ensemble-based predictions has also been crucial for the field of weather prediction. Not a mere technical advancement, but a veritable paradigm shift that brought more attention to the concepts of uncertainty and predictability @buizza_forecast_2015 @leutbecher_ensemble_2008. 

Advances have come with substantially increased computational and storage costs @bauer_quiet_2015, and despite the improvements in skill important shortcomings remain. NWP models still display systematic biases and ensemble systems struggle to appropriately represent uncertainty. 


=== Systematic and random errors
The technical limitations of NWP models explained in the previous paragraphs already give us an intuition of how prediction errors might originate and propagate in weather simulations. To deepen our understanding of the topic it is useful to focus on the structure of errors in slightly more detail, albeit still in an idealized manner. Let us start by defining the error $e$ as the difference between a prediction $hat(y)$ and corresponding true value (or observation) $y$,
$
e = y - hat(y).
$

Our central question is "What contributes to the error $e$?".
The primary distinction is between _systematic errors_ (or _bias_) and _random errors_. These can be seen as two components contributing to the total error, which we can then express as 
$
e = b + epsilon "    where" epsilon tilde cal(D)(0, sigma^2).
$

The systematic component $b$ is the part of the error that can be expected because it occurs consistently for any given situation. On the other hand, the random component $epsilon$ cannot be known precisely in advance and is described by a random variable drawn from a zero-mean distribution. The distinction between the two is usually done by appropriately aggregating several instances of $e$. For example, we can compute $b=EE[e]$ since we know that $EE[epsilon] = 0$. We can also derive $sigma^2="Var"(e - b)$ from that.

A crucial property of the error is that it is _flow dependent_: we can therefore consider $b$ and $sigma^2$ to depend on the state of the model $cal(S)$, and we can derive them by considering $EE[e | cal(S)]$. By aggregating errors in well-thought-out ways, it is possible to derive a great deal of information about the error structure conditional on the presence of some specific criteria in the state $cal(S)$. 

Analyzing different aggregations could, in principle, allow us to dissect the systematic error into many components or _sources_, although this becomes an especially hard challenge when multiple sources interact and the random error makes it harder to extract significant patterns. In general, analyzing the statistical properties of the error - more generally the joint distribution of forecasts and observation $p(hat(y), y)$ - to understand strengths and limitations of a prediction model is the goal of forecast verification, discussed in @2-forecast_verification. 


#figure(
  image("1-fig-lorenz.png"),
  caption: [
    Lorenz's system to illustrate the concept of flow-dependent predictability. The blue lines represent the phase space of the dynamical system, in other words all the possible states that the system can assume. The red lines represent trajectories starting from certain initial conditions. On the left is a predictable situation and on the right an unpredictable situation.
  ],
  placement: top,
) <ch1-fig-lorenz>

==== Sources of systematic error
Systematic errors arise from structural deficiencies of the NWP models. They are the result of inherent flaws in the model representation of certain processes. 
Firstly, model approximations and parameterizations are major sources of systematic errors. As mentioned in @1-limitations, the model resolution is an important limitation. Some weather phenomena cannot be represented directly by the model, because they arise from processes that occur at scales that are not resolved on the model's grid. They are instead represented through parameterization schemes that can introduce biases across various spatial and temporal scales. As an example, let us consider the parameterization of surface-atmosphere exchanges of water and energy through evapotranspiration. On a local scale, biases in the evapotranspiration schemes would contribute to systematic errors in temperature and humidity predictions. 
In practice, there is no way to parameterize all processes without any bias. Sensitivity analyses are carried out to understand the influence of certain parameters or coefficients on relevant weather phenomena (see #citet[@chinta_assessment_2021] for an example and an excellent review of the literature on the topic), and they are tuned accordingly. This also means that different national weather services might parameterize the same model in different ways according to their priorities. An important aspect to consider is that operational NWP models are subject to frequent evolution cycles, whereby slight changes to the parameterization schemes might alter the structure of systematic errors.


==== Predictability and random error
Contrary to the systematic error, there is no compelling reason for trying to understand the origin of random errors. In fact, we might even say that we have to purposely avoid it, since by definition we should accept that they are related to the chaotic nature of the atmosphere, and therefore any investigation based on a deterministic chain of cause and effect would be inconsequential. Nevertheless, understanding the properties of the random error is of crucial importance for another concept in weather forecasting, _predictability_ @krishnamurthy_predictability_2019. In essence, we want to understand how confident we can be of a model's predictions as we increase the forecasting horizon, which is strictly related to how fast the error, and in particular its random component, grows in the modelled dynamical system. The theoretical study of predictability has a long history, and a seminal paper by #citet[@lorenz_deterministic_1963] on the topic is now regarded among the most influential works in meteorology and beyond. The paper introduced what's known as the Lorenz system, a system of ordinary differential equations that despite its simplicity displayed many fundamental properties of chaotic systems such as the atmosphere. Among these properties is the _flow-dependency_ of the error and its growth, which we illustrate in @ch1-fig-lorenz. The idea depicted in this figure is that the predictability of a system depends on its current state: on the left, we see that all trajectories initialized at a specific state end up relatively close to each other. On the right, trajectories initialized for another state of the system diverge very quickly, making it more difficult to predict the future state of the system. 


=== Gap between model and reality <1-gap>

#figure(
  [#cetz.canvas({
    import cetz.plot: *
    import cetz.draw: *
    import cetz.decorations: flat-brace
    
    let tbox(body, stroke: 1pt, size: 11pt) = {
      box(align(center + horizon)[#text(size, body)], stroke: stroke, inset: 6pt, fill: white)
    }
    plot(
      size: (10,6),
      legend: "legend.inner-north-west",
      // legend-default-position: "legend.inner-north-west",
      x-tick-step: none, y-tick-step: none, 
      x-label: "Time", y-label: "Prediction error", 
      axis-style: "scientific-auto", 
      legend-style: (stroke: none), {

      
    add-fill-between(domain: (0, 0.5*calc.pi), style: (stroke: none),
     x => calc.sin(x) + 0.4,
     x => 0.0, label: text(size: 8pt, "Initial conditions"))
    
    add-fill-between(domain: (0, 0.5*calc.pi), style: (stroke: none),
     x => calc.sin(x*0.8) + 0.3 - x * 0.25,
     x => 0.0, label: text(size: 8pt, "Model physics"))
    
     add-fill-between(domain: (0, 0.5*calc.pi), style: (stroke: none),
     x => 0.3,
     x => 0.0, label: text(size: 8pt, "State representation"))
    })
    mark((-0.5 * calc.pi,0), (3.25 * calc.pi, 0), symbol: ">", fill: black)
    mark((0,1), (0, 6.3), symbol: ">", fill: black)

    flat-brace((3.3 * calc.pi, 0), (3.3*calc.pi, 1.3), flip: true, name: "brace1")
    flat-brace((3.3 * calc.pi, 1.3), (3.3*calc.pi, 6), flip: true, name: "brace2")

    content("brace2.east", tbox("State transition error\n(systematic and random)", stroke: none), anchor: "west", padding: 6pt)
    content("brace1.east", tbox("State representation error\n(systematic and random)", stroke: none), anchor: "west", padding: 6pt)
  }) #v(0.5cm)],
  caption: [Evolution of prediction error with time, relative to point-observation ground truth. At starting time, error results from a combination of deficiencies in initial conditions and state representation error. With time, error from initial conditions grows and shortcomings in the model's physics introduce additional error, plateauing at the error saturation point. State representation error, on the other hand, stays on average the same.],
  placement: top,
) <ch1-fig-error_evolution>


NWP models have a discrete state representation of the continuous real atmosphere. In the three-dimensional grid each cell is representative of the average real conditions found within its boundaries. However, it is not trivial to determine how the value of a grid cell translates to the real world. This becomes particularly problematic when we require predictions at specific points in space, as it often occurs for many meteorological and climatological applications. 
This gap is a primary contribution to the systematic error found at that particular location: a simple and illustrative example is the difference between the elevation of a model grid cell and the elevation on a corresponding point in the real world, resulting in a stable bias in temperature and pressure.
Furthermore, the model often does not represent a substantial part of the random variability at a particular location, resulting in an underestimation of the uncertainty. This is what happens, for instance, with wind, where sub-grid scale turbulence near the surface is not fully captured by the model.

Related to the gap between model and reality we can derive another useful concept to reason about model errors, in addition to the one discussed in the previous section. This concept distinguishes between what we will call _state representation error_ and _state transition error_, shown in @figure-stateerror. We can think of the latter as the part of the error arising from how the atmospheric state is evolved over time. In other words, it is only induced by deficiencies in the model during the transition from one state to the next, either systematic or random, which includes the computation of the initial conditions. Imagine a hypothetical scenario with no state transition error, where the model's physical representation induces no biases in the simulation over time. We can assume that the state at any grid cell correctly represents the average state of the real conditions within its boundaries. In such a situation, shown in the center of the figure, we will still have some systematic error when comparing the model with specific points in reality, for instance with in-situ measurements. This is the state representation error, exclusively related to how the state is represented to approximate the real world. 
In @ch1-fig-error_evolution we can also see this concept applied to error evolution. Because it is independent of the dynamic nature of the system, the relative importance of the state support error is larger compared to the state transition error for short forecast horizons and particularly for the model analysis. The opposite is true for the state transition error, which grows as the model simulation advances. Similarly, we can say that, at any given simulation time, the state transition error depends on the previous states or the simulation trajectory, whereas the state support error only depends on the state itself, although it will stay constant on average.

The conceptual distinction we have just introduced also relates to the question of what we consider _ground truth_. 


#figure(
  [#cetz.canvas({
    import cetz.draw: *

    let tbox(body, stroke: 1pt, size: 11pt) = {
      box(align(center + horizon)[#text(size, body)], stroke: stroke, inset: 6pt, fill: white)
    }
    // polygons with annotations
    on-xz({
     rect((0, 0), (2, 2), stroke: blue, name: "rect1")
    })
    content("rect1.south", anchor: "north", padding: 10pt, "Grid cell prediction")
    
    on-xz({
     rect((6, 0), (8, 2), stroke: blue, fill: rgb(200, 0, 0, 40), name: "rect2")
    line(("rect1.east", 10%, "rect2.west"), ("rect1.east", 90%, "rect2.west"), stroke: 1pt, mark: (symbol: ">"), fill: black)
    })
    content("rect2.south", anchor: "north", padding: (y: 10pt, x: 10pt), "Average truth within \n the cell (hypothetical)")

    
    on-xz({
     rect((12, 0), (14, 2), stroke: blue, name: "rect3")
    line(("rect2.east", 10%, "rect3.west"), ("rect2.east", 90%, "rect3.west"), stroke: 1pt, mark: (symbol: ">"), fill: black, name: "line2")
     circle("rect3.center", anchor: "center", fill: red, radius: 0.2)
    })
    content("rect3.south", anchor: "north", padding: 10pt, "Truth at specific \n point within the cell")

    let x0 = 1.0
    content((x0, 2), anchor: "south-west", tbox("Systematic errors \n resulting from \n model physics", size: 8pt), name: "box1")
    content((x0+3, 2), anchor: "south-west", tbox("Random errors \n resulting from \n large scale noise", size: 8pt), name: "box2")
    content((x0+2.5, 4), tbox("State transition errors", stroke: none))

    line("box1.south", (4.5,1))
    line("box2.south", (4.5,1), name: "line1")
    line("line1.end", (4.5,0.5))

    x0 = x0 + 7.5
    content((x0, 2), anchor: "south-west", tbox("Systematic errors \n resulting from \n representation gap", size: 8pt), name: "box1")
    content((x0+3, 2), anchor: "south-west", tbox("Random errors \n resulting from \n sub-grid scale noise", size: 8pt), name: "box2")
    content((x0+3, 4), tbox("State representation errors", stroke: none))
    
    line("box1.south", (6+4.5,1))
    line("box2.south", (6+4.5,1), name: "line1")
    line("line1.end", (6+4.5,0.5))
  }) #v(0.5cm)],
  caption: [
    Illustration of the conceptual difference between state transition and representation errors. Adapted from #cite(<owens_ecmwf_2018>, form: "prose").
  ],
  placement: top
) <figure-stateerror>


=== Data-driven weather prediction <1-ddwp>

In recent years, machine learning (ML) methods have been successfully applied to an extremely wide range of scientific and engineering challenges. Deep learning, in particular, has shown to be capable of modelling complex and high-dimensional data in domains such as computer vision and neural language processing, attracting the interest of the weather modelling community. With the central question being whether modelling the atmosphere can be learned purely from data, some early exploratory work was conducted by e.g. #cite(<dueben_challenges_2018>, form: "prose") and #cite(<scher_toward_2018>, form: "prose"), and generated some confidence among researcher about the feasibility of such approaches. Shortly after, some studies presented the first data-driven weather prediction models that had comparable performance to traditional NWP models while being orders of magnitude cheaper to compute @rasp_datadriven_2021@weyn_improving_2020@clare_combining_2021. From there, we are witnessing intense efforts from both research institutions and large tech companies to rapidly improve these data-driven models, at first focusing on deterministic prediction @pathak_fourcastnet_2022@keisler_forecasting_2022@lam_learning_2023@chen_fuxi_2023@bi_accurate_2023@nguyen_scaling_2023 and then extending to ensemble forecasts @price_gencast_2024@lang_aifs_2024 and probabilistic forecasts @andrychowicz_deep_2023[predicting quantiles]. End-to-end data-driven approaches, including data-assimilation, have also been successfully implemented @huang_diffda_2024@vaughan_aardvark_2024. Furthermore, following the work of #cite(<oskarsson_graph-based_2023>, form: "prose"), limited-area regional models are under active development, with a particular interest from national weather services, including MeteoSwiss.

Despite some drawbacks, notably in terms of overly smooth forecasts and lack of detail in the representation of high-impact events, data-driven models are now outperforming NWP models in many of the standard verification metrics @rasp_weatherbench_2024@bouallegue_rise_2024@charlton-perez_ai_2024 and are showing enormous potential, considering that research in this area is still in a premature phase. It is not unreasonable to say that in the matter of a couple of years we have entered a new era for the field of weather prediction. What does this mean in the context of this work? #TODO[Find a better way to end the paragraph and transition to the next.]

Data-driven prediction models are typically trained with gridded analysis or reanalysis data as targets, and, as for any ML model, they can only be as good as the training data itself. Furthermore, they are built to reflect the structure of the training data, which is why the core component of any data-driven model still resembles a grid that represents the discretized atmospheric state on which the model operates. Of course, the grid representation in a data-driven model differs substantially from an NWP model, but they share common limitations. Specifically, the level of representational detail of the atmospheric state in data-driven models cannot surpass the one found in the training data. This consideration is relevant in light of what we discussed in @1-gap. The main reason why data-driven models may outperform NWP models is that they are specifically trained to reduce the state transition error. However, they have the same limitations regarding state representation error, that is, the gap between the model and reality is still there.

In practice, this means that we still require some additional components to map the model's discrete state representation to the real world. A potential advantage of data-driven approaches is that, thanks to their flexibility, this component could be easily integrated into the models themselves instead of being a separate step, as is the case for NWP models. As explained in @1-postprocessing, it could be possible to derive useful features from the model's state even from its internal compressed representation.

Another approach to reduce the representational gap is to improve the training data itself and therefore allow a more detailed representation of the atmospheric state, especially at the surface. This can be done, for instance, by running very high-resolution simulations, but it can be prohibitively expensive. Alternatively, one could employ ML techniques to improve the NWP model analyses, as we will see in #ref(<chap-4>, supplement: [Chapter]).


== Improving the quality and usefulness of weather predictions <1-usefulness>

Without diminishing the important role of meteorology in gaining a scientific understanding of the natural world, it is safe to say that meteorology is primarily an intensely practical discipline. As noted by #citet[], a major achievement of the weather enterprise has been the strong symbiosis between science and operations, which is fundamental for producing and disseminating weather and climate information in the form of knowledge, data and services. Many different components - individuals and organizations with diverse categories of activity - work in coordination to succeed in such endeavor, and one way to describe the full picture of the weather enterprise is through the lens of the meteorological value chain, which is illustrated by @2-fig-weatherchain. In essence, the meteorological value chain is the combination of activities in the weather enterprise linking scientific and technical advancements to societal and economic demands. Here, we have divided the value chain into two main groups, a simplification that will help us clearly identify the scope of this work. On the left, comprising of the collection of observations and the development and operation of weather models is part of the value chain that is responsible for creating the foundational knowledge and data bases for weather and climate information. Activities in this domain are closer to the more scientific and technical aspects of meteorology, and how that information will be used downstream is not a primary concern. However, as explained in the previous section, the raw output of NWP or DDWP models has important shortcomings, namely the presence of systematic biases and the incorrect representation of uncertainty. Furthermore, the amount of information provided by those raw outputs might be insufficient for some applications or unnecessarily large for others, which respectively prompts for some augmentation or distillation in what are often referred to as _post-processing_ steps.

The second group consists of all the activities intended to use the foundational data and knowledge bases to develop products and services that respond to the direct needs of society and the economy by refining and optimizing information. It represents, in a nutshell, the broader scope of this thesis. 

We may further divide these activities into three main categories. Generic weather forecasts are user-agnostic products usually disseminated to the public through digital communication systems such as smartphone applications and the media. They are typically produced for common weather parameters such as temperature, wind, precipitation or cloud cover. Tailored products, on the other hand, are user-specific and they focus on specific aspects of the weather, such as a particular region or variables that are relevant for a certain application. Integrated services utilize meteorological data in combination with models and datasets that focus on non-meteorological factors. In all three categories, the underlying goal is to improve the end-users' decision-making process for their economic benefit and well-being.


#figure(
  [#cetz.canvas({
    import cetz.draw: *
  
    let cx = 0; let cy = 0
    let lg = rgb(205, 225, 205)
    let g = rgb(130, 179, 102)
    let mk = (symbol: "stealth", fill: black, scale: 1.5)
    let mkg = (symbol: "stealth", fill: g, scale: 1.4)

    let dx = 0.6

    let tbox(body, stroke: 1pt, size: 11pt) = {
      box(align(center + horizon)[#text(size, body)], stroke: stroke, inset: 6pt, fill: white)
    }
    
    rect((cx - dx, cy), (-4.3, 4), anchor: "center", name: "rect1")
    content(("rect1.south", 30%, "rect1.north"), tbox("Weather \n modelling"))
    content(("rect1.south", 70%, "rect1.north"), tbox("Observations"))
    
    rect((cx + dx, cy), (4.3, 4), anchor: "center", stroke: (paint: g, thickness: 0.5pt), fill: lg, name: "rect2")
    content(("rect2.south", 80%, "rect2.north"), tbox("Forecasts")) 
    content(("rect2.south", 20%, "rect2.north"), tbox("Integrated \n services"))
    content(("rect2.south", 53%, "rect2.north"), tbox("Tailored products"))
    
    line("rect1", "rect2", mark: mkg, stroke: g)

    
    content((-5.5, 2), tbox([Science \ and \ technology], stroke: none, size: 13pt), anchor: "east", name: "text1")
    line("text1.east", "rect1", mark: mk)
    content((5.5, 2), tbox([Society \ and \ economy], stroke: none, size: 13pt), anchor: "west", name: "text2")
    line("rect2.east", "text2.west", mark: mkg, stroke: g)
    
  }) #v(0.5cm)],
  caption: [Illustration of the meteorological value chain bringing scientific and technological advances to the social and economic benefit. At the same time, these advances are driven by the demands of the customers. Adapted from #citet[thorpe_global_2022]. Parts of the diagram colored in green are related to the optimization of weather information and thus the broader scope of this thesis.],
  placement: top,
) <2-fig-weatherchain>



=== The value of a weather forecast <1-value> 
We have learned that, within the weather enterprise, significant resources go specifically into maximizing the value of weather information by refining, augmenting and combining raw data. This is essentially all activities within the green box in @2-fig-weatherchain. In this diagram, the right-hand green arrow represents the relationship between product creators and users. It is not simply the delivery of a forecast product, but represents a wider exchange of information preceding and following said delivery, such as requirement specifications. We can therefore say that maximizing the value of weather information is actually a joint effort of product creators and users. 

In this context, understanding the economic value of weather forecasts is of paramount importance, as it plays a crucial role in legitimizing research and gives guidance on the objectives that should be pursued when optimizing forecasts. Only when we know what we are looking for can we develop and choose the right tools to evaluate how close we are getting to our goals, as shown in @2-forecast_verification. 

The fundamental principle for determining the value of weather information is understanding its effect on decision-making processes. Decision-making models are developed for this purpose. The available literature focuses more specifically on the _economic_ value of weather information, because it is generally easier to express in quantifiable (financial) terms, and the most widely used approach is that of the cost-loss analysis as described in @murphy_forecast_1997. 

#citet[@zhu_economic_2002] discusses the economic value of ensemble-based weather forecasts

Over the years, several models #citet[@millner_what_2009] proposed a model that relaxes some assumptions about users' behavior, notably accepting that they might have an incomplete or erroneous statistical understanding of forecast performance, and show that this generally leads to a not-fully-realized economic value of the forecast. This goes to reinforce the idea that exchanging information between forecasters and users, for example via education initiatives, can be particularly beneficial. 


=== Specialization or generalization
Even after having determined what constitutes a valuable forecast, there are still some high-level principles to consider before looking into the more technical or scientific aspects of optimizing weather information. It is often the case that the set of desired properties for a new product or service are relatively simple to obtain if taken individually, yet become very challenging when taken together, as discussed in @1-sota.

This prompts for discussions about whether it is better to develop several specialized products satisfying different needs or to develop a single or few general products satisfying all needs simultaneously #footnote("During several meetings and workshops that I attended at MeteoSwiss, the discussion around this tradeoff was among the most recurring topics, and there is no general consensus among experts."). 

Typically, specialized products perform better for their specific task
// Among these is the question of whether we should develop 

== Statistical post-processing of direct model output <1-postprocessing>
Post-processing is a term used to define the broad range of activities aimed at optimizing information from weather models for specific applications, by applying additional processing steps to the direct model output (DMO). A subset of these activities is called _statistical post-processing_, and it specifically focuses on addressing the shortcomings of weather models discussed in @1-limitations. It does so by taking advantage of the statistical relationship between DMO forecasts and observations - or equivalently, in some cases, the statistical properties of the errors - determined from historical data. In essence, it is primarily about identifying and correcting systematic errors and, in the case of ensemble forecasting, incorrect representations of uncertainty in weather models @wilks_statistical_2019@vannitsem_statistical_2018.

One of the shortcomings of weather models is their discrete state representation of the atmosphere, meaning that DMO only provides information about the atmosphere for a limited number of fixed locations, times and meteorological parameters. By leveraging the statistical relationship between the discrete atmospheric state in the DMO and observations in the continuous real atmosphere, statistical post-processing can also be seen as a way to expand the state space of a model @stephenson_forecast_2005, allowing to issue forecasts at arbitrary spatial and temporal locations and for new meteorological parameters. 


#figure([
  #cetz.canvas({
    import cetz.draw: *
    
    let tbox(body, stroke: 1pt, size: 11pt) = {
      box(align(center + horizon)[#text(size, body)], stroke: stroke, inset: 6pt, fill: white)
    }
    let mk = (symbol: "stealth", fill: black, scale: 1.5)
    
    let x0 = 0.0; let y0 = 1.5; let dx = 6.0; let dy = 2.0
    
    rect((x0, y0), (x0 + dx, y0+dy), name: "box1")
    content("box1.center", align(center + horizon)[Discrete atmospheric state \ (internal or DMO)])

    y0 = -1.5
    rect((x0, y0), (x0 + dx, y0+dy), name: "box2")
    content("box2.center", align(center + horizon)[Auxiliary features \ (static or dynamic)])

    circle((10.0, 1.0), radius: 1.8, name: "circle")
    content("circle.center", align(center + horizon)[Statistical \ post-processing \ model])

    line("box1.east", (7.5, 1.0), "circle.west")
    line("box2.east", (7.5, 1.0), "circle.west")

    
    
  }) #v(0.5cm)],
  caption: [
    A general, conceptual diagram for statistical post-processing of weather forecasts. Predictors are either features extracted from a model's predicted discrete atmospheric state or auxiliary covariates. They are statistically related to the predictands, which can be any observable meteorological quantity, via the post-processing model.
  ],
  placement: top,
) <fig-postprocessing>

In practice, statistical post-processing can be performed in many different ways, each one with its strengths and weaknesses, as demonstrated by the rich literature on the topic @vannitsem_statistical_2021. We will discuss the main methodological aspects characterizing the variety of approaches later in this section, but we can first attempt to give a general and universal definition of statistical post-processing.

@fig-postprocessing illustrates its main components. The main source of input is the discrete atmospheric state of a model. This is usually the DMO, but in principle it could be any intermediate field or internal state representation, notably in the case of DDWP models. Predictors are derived from the model state in various ways depending on the task at hand, which represents the step of _feature engineering_ described in @2-feature_selection. In addition to predictors derived from the weather model predictions, other auxiliary features can be used. These can be static in time, such as those related to geomorphological information, or dynamic such as external factors related to seasonality or diurnal cycle. Generally, these auxiliary factors are covariates in the sense that they have alone some predictive power, but it is also common to include factors that are not directly related to the predictand but rather moderate how other covariates are related to it in different situations. Predictands can be any observable meteorological quantity, with the most common case being in-situ observations from the surface. Statistical post-processing models relate predictors to predictands. 

#TODO[Something is missing.]



=== Review and methodological distinctions <1-review>

We will not extensively review the statistical post-processing literature here - we refer readers to @vannitsem_statistical_2018@vannitsem_statistical_2021 for a comprehensive discussion - and instead focus on describing the most relevant methodological aspects to consider. The aspects described below are not necessarily orthogonal and may in some cases intersect, but they can be regarded as useful abstractions to organize the great variety of existing methodologies.

==== Point- and grid-based post-processing

#figure([
  #cetz.canvas({
    import cetz.draw: rect, content
    import cetz.plot
    import cetz.palette


    
    let grid(size) = {
        pattern(size: (size, size))[
        #place(line(start: (0%, 0%), end: (0%, 100%)))
        #place(line(start: (0%, 0%), end: (100%, 0%)))
      ]
    }

    rect((0,0), (2,2), fill: grid(0.5cm))
    rect((3,0), (5, 2), fill: grid(0.25cm))
    content((0,3),[a)])

    
    
    
  })
],
  caption: [Ciao.],
  placement: top,

)

A potential source of confusion for those approaching the domain of statistical post-processing is concerning the actual target of the regression task. Traditionally, statistical post-processing has been applied to relate the weather model predictions to observed meteorological quantities observed at surface stations. In this case, the target is a point that lies in the continuous real atmosphere.

More recently, with the advent of machine learning approaches, other models have been proposed (see e.g. #citet[@gronquist_deep_2021]) that instead instead map the discrete model state to another discrete state that is usually more refined, thus reducing the representational error, while also removing eventual biases and enhancing the quantification of uncertainty, which addresses the state transition error (see @1-gap). To do so, NWP model analysis or reanalysis data is used as target, and 
convolutional neural networks can be employed, taking inspiration from computer vision approaches. We refer to this models as _grid-based_.

While both can be considered statistical post-processing approaches, they have fundamentally different goals. Grid-based post-processing is confined to the model space and its main purpose is to improve the performance of weather models for small computational costs, potentially allowing a strategic reallocation of resources such as lowering the number of ensemble members to increase a model resolution @gronquist_deep_2021. However, it does not directly pertain to the creation of an end product that is optimized with respect to real world observed conditions. This can be only achieved with point-based post-processing. These differences are rarely made explicit in the existing literature, which we argue can be problematic when comparing the evaluation of statistical post-processing approaches, especially for individuals who are new to the field. When not stated otherwise, for the remainder of this work we will implicitly assume that statistical post-processing refers to the point-based approach.



==== Local and global post-processing
#figure([
  #cetz.canvas({
    import cetz.draw: rect, content
    import cetz.plot
    import cetz.palette


    
    let grid(size) = {
        pattern(size: (size, size))[
        #place(line(start: (0%, 0%), end: (0%, 100%)))
        #place(line(start: (0%, 0%), end: (100%, 0%)))
      ]
    }

    rect((0,0), (2,2), fill: grid(0.5cm))

    rect((3,0), (5, 2), fill: grid(0.25cm))

    content((0,3),[a)])
    
    // plot.plot(size: (3,3), {
      
    //   let z(x, y) = {
    //     (1 - x/2 + calc.pow(x,5) + calc.pow(y,3)) * calc.exp(-(x*x) - (y*y))
    //   }
      
    //   plot.add-contour(
    //     x-domain: (-2, 3), y-domain: (-3, 3),
    //     z, z: (.1, .4, .7), fill: true
    //     )
    //   }
    // )

    
  })
  // #set page(margin: 0pt)
],
  
  // #let grid = pattern(size: (0.2cm, 0.2cm))[
  //   #place(line(start: (0%, 0%), end: (0%, 100%)))
  //   #place(line(start: (0%, 0%), end: (100%, 0%)))
  // ]
  // #rect(fill: grid, width: 2cm, height: 2cm)],
  
  caption: [Monecian],
  placement: top,

)
For many years, the most common approach to statistical post-processing has been to consider target locations independently, and to develop one statistical model for each specific location, hence the term _local_ post-processing. Evidently, the disadvantage is that this approach can only be applied where observations are available and can therefore not be generalized in space. On the other hand, developing one model for each station makes it possible to adapt and specialize the model to the specifics of the task at hand. For instance, one could easily account for microscale effects that are particular to a location without necessarily using contextual information about the station. From a practical perspective, another important point is that a local post-processing system for a large number of stations can become very complex and difficult to maintain#footnote("As reported via personal communication by members of the Deutsche Wetterdienst (DWD) working on MOSMIX, a local post-processing system producing forecasts for more than 5000 locations worldwide."), particularly if some manual tuning is needed for each location. Recently, #citet[@rasp_neural_2018] proposed a way to perform local post-processing while working with a single model, and is the same approach adopted in our work presented in  @chap-3.

The alternative method, often called _global_ post-processing, consists in developing a system capable of issuing forecasts at any point in space. To this end, several approaches can be used. A first approach is simply applying a spatial interpolation method to well-calibrated forecasts at stations @taillardat_research_2020. Alternatively, one can instead learn how to interpolate parameters (clearly, this concept only works on models with few parameters, such as the coefficients of multiple linear regression) of local post-processing models to any given location to create a new "virtual" local model. 
// The interpolation is usually based on geographical distance and geomorphological information, but it can also in principle depend on the weather state itself, in other words it can be flow-dependent. 
The second approach is to develop a single model that learns to use auxiliary information, such as geomorphological features or information about the land cover, as predictors. We will discuss more about the choice of these predictors in @2-feature_selection. An important aspect of the global approach is that its effectiveness is highly dependent on the number of stations available in the historical archive used to develop a model, as well as the representativity of those stations with respect to the full range of potential conditions.
Another strategy is to use the auxiliary information to normalize the training data with respect to spatial heterogeneities, to create "potential values", learn the statistical correction and then transform back again using the auxiliary information @dabernig_spatial_2017.

==== Distributional assumptions <1-distributional>

From a statistical standpoint, the existing techniques in probabilistic post-processing are broadly divided into two categories: those that presume the predictive distribution fits within a known class of probability distributions (parametric or distribution-based methods) and those that do not (nonparametric or distribution-free methods). 

Methods with distribution-based assumptions require learning the relationship between the input predictors and the parameters of the output distribution. For instance, if we assume the target $y$ to follow a Gaussian distribution, $y tilde cal(N)(mu, sigma^2)$, we will fit a model that uses the inputs to predict $mu$ and $sigma^2$. These methods can be computationally efficient by leveraging well-established statistical theories and models, simplifying the mathematical treatment and implementation of the models. Parametric distributions come with analytic formulas for the computation of the probability density function, allowing to optimize models using the maximum likelihood approach. Alternatively, for many types of distributions, model fitting can also be achieved via efficiently computed closed-form expressions of the CRPS @taillardat_calibrated_2016.  Cumulative distribution functions and quantile functions are also easily derived.
If the assumptions about the output distribution are valid, these methods can provide highly accurate predictions. However, selecting a suitable parametric family that describes the target variable is not trivial and remains an important limitation.

Methods without parametric assumptions about the target essentially work by approximating the forecast distribution in various ways. A common approach is to use quantile regression methods in which models are trained to predict a selected set of quantiles. Alternatively, one can model the predictive density of the forecast distribution by a histogram, essentially reframing a regression problem into a classification problem where each bin of the histogram represents a category. Approaches using various types of decision-tree models have also been proposed. Analog-based methods also fall under the category of distribution-free approaches, as well as member-by-member post-processing. Indeed, there is a large variety of methods to approximate the output distribution.
These methods come with the main advantage of being more flexible than their parametric counterpart: they can capture patterns that do not fit traditional statistical distributions, such as distributions with specific skewness, heavy tails or multimodal patterns. On the other hand, these methods often require significantly more computational resources for training and prediction, their flexibility also means that they are more prone to overfitting if not properly regularized, and they typically require large amounts of data to be trained effectively.

==== Modelling the functional relation

Distributional assumptions concern the way we represent the output predictive distribution, but they do not explain how in practice we derive it from the inputs. This is another aspect, pertaining to the functional relation between inputs and outputs. The first applications of statistical post-processing were based on methods such as multiple linear regression @glahn_use_1972. Over the years, more complex models have been proposed, with most recent methodologies based around ML @vannitsem_statistical_2021.

In some cases, the modelling choice for this relation also comes with a distributional assumption by design. For instance, random forests or analog-based models naturally output an approximation of the predictive distribution via a finite set of samples. In other cases, distributional assumptions are completely distinct from the modelling choice for the functional relation. 


// ==== Training schemes
// @lang_remember_2020
// @demaeyer_correcting_2020

// #let review_table = figure([
//   #set text(size: 9pt)
//   #table(
//     columns: 4,
//     table.header[*Publication*][*Model type*][*Method type*][*Distributional assumptions*],
//     [@glahn_use_1972], [Multiple linear regression], [point, local], [-]
//   )
// ])

// #review_table

// @glahn_use_1972: model output statistics (MOS)

// lemcke_model_1988: logistic regression, not just linear regression

// marzban_neural_2003: neural network for post-processing (small models and one for each station)




// @raftery_using_2005: Bayesian model averaging (BMA)

// @gneiting_calibrated_2005: ensemble model output statistics (EMOS) with CRPS

// Hamill TM, Whitaker JS, 2006: analog method

// Wilks and Hamil, 2007: comparison of different EMOS systems

// linear variance calibration (Kolczynski), consensus forecasts

// Greybush et al 2008: consensus forecasts, regime-dependency 

// Delle Monache 2013: analog method 

// @hemri_trends_2014: statistical post-processing methods are still useful despite advances in NWP, 

// Hamill TM, Scheurer, 2015: analog method

// @taillardat_calibrated_2016: quantile regression forest models for post-processing

// @baran_mixture_2016: mixture EMOS model, wind, finding the right distribution for a variable

// Messner et al., 2017 (2016?): boosting for ensemble prediction -> "avoiding overfitting", gradient boosting

// // @rasp_deep_2018: neural networks for postprocessing

// Taillardat et al 2019: forest-based semiparametric precipitation post-processing

// Lang et al 2019: bivariate gaussian wind vectors, additive parametric model

// Simon et al 2019: additive parametric model 

// Schlosser et al 2019: distributional regression forest, precipitation

// @taillardat_research_2020: quantile regression

// Hopson, TM et al., 2010: quantile regression

// @bremnes_ensemble_2020: quantile function regression with NNs and Bernstein polynomials

// @gronquist_deep_2021: CNNs for "post-processing" (ground truth is analysis data)

// @kirkwood_framework_2021: QRF + interpolation to output a full PDF

=== State of the art and current challenges <1-sota>
In recent years, the most important trend observed in the field of post-processing is without doubt the widespread adoption of ML methods. Many of these methods have existed for decades already, including applications to statistical post-processing @marzban_neural_2003[for instance], but they could show their true potential only in recent years thanks to a combination of computational advances and availability of large datasets for training them. Among ML methods, neural networks are standing out as a particularly promising way forward thanks to their flexibility in handling diverse, heterogeneous data.

The blooming of statistical post-processing techniques over the last years has called for ways to enable a fair, quantitative comparison between them @vannitsem_statistical_2021@haupt_towards_2021. In a joint effort, several experts from European national weather services developed a platform aiming to do just that @demaeyer_euppbench_2023, providing a benchmark dataset for post-processing tasks (both point-based and grid-based)

In their review, #citet[@vannitsem_statistical_2021] identified three main methodological challenges in the field: the preservation of the multivariate dependence structure in the post-processed predictions, both in terms of spatio-temporal covariance structure and physical consistency between meteorological parameters; the blending of multiple sources of information such as multiple forecasts with different resolutions and forecasting horizons and observational data such as in-situ measurements or remote-sensed data; the issue of NWP model changes, which creates heterogeneities in the training dataset. We further discuss the first two as they have been the subject of two studies during this project, presented in @chap-3[Chapter] and @chap-4[Chapter].  

==== Spatio-temporal and inter-variable consistency
Accurately describing the state of the weather requires including spatial, temporal, and multi-variable information. It is more than just providing information about single variables at a specific locations and times, or the _margins_ of the weather state: there is also useful information in their joint variability or in other words the _dependence structure_ of the weather state. 

Traditionally, however, statistical post-processing approaches assumed independence of different forecast margins, that is, they only focused on optimizing each margin individually in an univariate sense. As a consequence, the multivariate dependence structure of the predictions that is typically present in NWP forecasts is lost after applying statistical corrections. This is true for both deterministic post-processing approaches, where the resulting predictions are often overly smooth, and probabilistic post-processing approaches, especially those relying on some form of sampling which results in uncorrelated variability between different margins.

This loss of information is problematic for several downstream applications, particularly those requiring realistic forecast scenarios. Coherent scenarios allow users to aggregate forecasts over spatial domains of interest or temporal windows in a meaningful way, as well as to consider the interaction between different variables. From this, it is possible to estimate the probability of occurrence of arbitrarily defined events, for instance the occurrence of a certain amount of precipitation over a hydrological catchment over 24 hours, or the co-occurrence of high temperature and humidity. To address this issue, several methods have been proposed that aim to conserve or re-establish the dependence structure of the predictions. Excellent reviews and comparisons of such approaches, in the case of probabilistic post-processing, can be found in #citet[@schefzik_chapter_2018], #citet[@lerch_simulation-based_2020] or #citet[@lakatos_comparison_2023]. From these contributions we can draw a few very general observations.

A common thread of many existing approaches is the use of empirical copulas, either coming from historical observations @clark_schaake_2004 or from a raw NWP ensemble @schefzik_ensemble_2017. These non-parametric approaches are generally preferred over parametric counterparts because they are more generally applicable, as they do not require assumptions about the dependence structure and they can be applied to very high-dimensional problems, such as the multivariate prediction of multiple variables over a dense grid and a long temporal window. As noted in #citet[@schefzik_chapter_2018] parametric approaches have been applied with success to model either spatial or temporal or inter-variable dependencies, but never to all of them simultaneously. Furthermore, existing approaches to model spatio-temporal dependencies either rely on a pre-existing NWP raw ensemble @berrocal_combining_2007 or on somewhat simplistic covariance models @feldmann_spatial_2015. The latter approach shares some similarities to the methodology presented in @chap-4[Chapter], which in turn shows that more complex covariance models can be used.

Finally, an important point is that the dependence structure of probabilistic post-processing methods is usually modelled in a _statistical_ sense. That is, we focus on reproducing the statistical properties of the dependencies in the multivariate predictive distributions. However, these statistical dependencies are consequences of precise physical principles, and directly incorporating those principles in our models can be viewed as an alternative approach to achieve a similar yet stronger type of consistency: _physical_ consistency instead of statistical. We bring this concept to the field of post-processing with our work presented in @chap-3[Chapter].


==== Blending diverse sources of information <1-blending>
Information about the atmospheric state comes from a variety of sources, including both observational and modelled data. Observational data can come in several formats, depending on the quantity of interest and the measuring system used. Surface weather stations provide data as single point observations; aircrafts and weather balloons radiosondes, as well as some types of satellites, produce spatio-temporal trajectories; remote-sensed measurements such as those from satellites or radars can be represented on regular grids.
Meanwhile, modelled information from NWP typically exists on grid of varying spatial and temporal resolutions, and in the case of weather forecasts it might cover forecasting horizons of different lengths into the future. Overall, the way the atmospheric state is represented in the data at our disposal is quite heterogeneous. 

Methodologies attempting to optimally combine heterogeneous meteorological data to improve the accuracy of our prediction systems are known as _blending_ techniques. #citet[@vannitsem_statistical_2021] reviews the most prominent approaches. 

Traditionally, a blending scheme is a system that explicitely computes and assigns weights to different input sources, ideally in a dynamic way that is dependent on scale and atmospheric state -- being therefore flow-dependent. The weighting is then applied either in the probabilistic space, with exceedence probabilities or quantiles @kirkwood_framework_2021@roberts_improver_2023[for instance], in the physical space, with actual realizations @haiden_integrated_2011[for instance] or decompositions thereof @imhoff_scaledependent_2023[for instance].  Currently, a main challenge concerning blending is the preservation of the coherence structure in the blended product, in the same way discussed in the previous sub-section. Furthermore, the review of the literature shows that while a great deal of work exists regarding the blending of radar observations with NWP data -- one might say it is a field on itself -- or the blending of multiple NWP models, little to no research has been conducted concerning the blending of observational data from surface stations with NWP models, in a probabilistic forecasting context. The work presented in @chap-4[Chapter] studies methods to address both of these issues under the common concept of covariance modelling.

// A particularly attractive way to pursue is generative ML. 
// For instance, #citet[@leinonen_latent_2023] proposes using latent diffusion models for precipitation nowcasting and envisages the inclusion of NWP data as additional predictors, which would effectively result in an implicit blending system. This type of approaches, however, are harder to implement when the ground truth




// ==== Operational implementation <1-operational>
// #lorem(200)


== Research scope <1-scope>


The scope of this thesis is the optimization of weather information, a somewhat broader scope that includes, but is not limited to, the statistical post-processing discussed in @1-postprocessing. 

// Our work initiated during a surge of interest in ML methodologies for statistical post-processing of weather forecasts.


Throughout this thesis, we have used NWP models used operationally at MeteoSwiss as sources of raw forecasts for our experiments, such as COSMO-1E was used as the ensemble predicton system providing raw model outputs for our experiments. The model produces forecasts for the domain shown in blue in @1-fig-domains, and it has a horizontal resolution of 1.1 km. It has been used operationally at MeteoSwiss from #text("missing", fill: red) until July 2024.

Th


#figure(
  image("1-fig-domains.png", width: 90%),
  caption: [
    The COSMO models domain (blue line) and the swiss radar domain (red line). #sym.copyright\Stadia Maps #sym.copyright\Stamen Design.
  ],
  placement: top,
) <1-fig-domains>



== Thesis outline <1-outline>

The remainder of this thesis is organized into the following chapters.
@chap-2[Chapter] aims to summarize the technical knowledge required to successfully optimize data from weather models, with both theoretical reviews of certain key concepts as well as more practical discussions of the tools. Specifically, it will focus on three main topics -- forecast verification, machine learning and feature selection -- and will do so from the lenses of our research scope outlined in the previous section.

@chap-3[Chapter] presents the concept of integrating physical principles into statistical post-processing models to achieve physically consistent forecasts, therefore addressing one of the challenges , detailing the data and methods used, and discussing the results with a focus on predictive performance, physical consistency, and robustness to data scarcity. @chap-4[Chapter] focuses on improving weather model outputs at very high spatial resolutions, specifically at sub-kilometer scales. It explains the data and methods used, including neural network-based post-processing and Gaussian Processes regression, and provides a comprehensive analysis of experiments, results, and case studies. Finally, @chap-5[Chapter] summarizes the main findings, discusses broader implications, and provides an outlook on future research and potential applications of the developed methods. This structure ensures a logical progression from foundational methodologies to practical applications and implications in weather information optimization.