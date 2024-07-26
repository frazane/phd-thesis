// ---------------------------------------------
// Helper functions
// ---------------------------------------------
#let citet = (citation) => {
  set cite(form: "prose")
  citation
}


#let TODO(body, color: yellow) = {
  rect(
    width: 100%,
    radius: 3pt,
    stroke: 0.5pt,
    fill: color,
  )[
    #body
  ]
}