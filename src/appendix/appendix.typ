
// #heading("Appendix", numbering: none)

// #counter(heading).update(0)
// #set heading(numbering: "A.1")


= Related research
Also known as side quests. 

== Flexible probabilistic regression with parametric transformations
Show experiments with Bernstein normalizing flow.

= Contributions to open source scientific software

== `scoringrules` <appendix-scoringrules>

As shown in @2-forecast_verification, forecast verification is a fundamental aspect of meteorology and climatology. 

In the world of open-source software, the `scoringRules` library @jordan_evaluating_2019 available for the `R` programming language provides comprehensive functionality for comparative evaluation of probabilistic models based on proper scoring rules, covering a wide range of situations in applied work. The absence of a library of equivalent quality and coverage for the `Python` programming language prompted the creation of `scoringrules`.



The library is developed with three main design principles in mind.


==== Didactic approach
A recurring issue in specialized scientific software is assuming that users are already familiar with the more theoretical aspects underlying its implementation and usage. This limits the accessibility of a library to a wide user-base, and it can also mean that the library itself is used improperly. To avoid this, `scoringrules` is developed in a way that facilitates people that are approaching the field of forecast verification. In the library documentation, explanations, mathematical formulas and references are given for each metric. In the code, doc-strings are informative and the code itself is as close as possible to the mathematical formulas, so as to help transitioning between the two languages.


==== Performance
`scoringrules`

==== O



It can not only provide a measure to evaluate the quality of predictions _a posteriori_, but it can also serve as a 


#figure(
  ```python
  >>> import scoringrule as sr
  >>> sr.crps_normal(10.2, 12.3, 1.2)
  array(0.1, dtype=float32)
  ```,
  caption: [Example usage of `scoringrules` to compute the CRPS for the normal distribution.],
  supplement: "Code"
)

== `mlpp` framework <appendix-mlpp>
`mlpp` is a ML-based post-processing library used operationally at MeteoSwiss. 


== `GPJax`
...


= Supplemental notes
== Heterogeneous measuring conditions: a problem or an opportunity?