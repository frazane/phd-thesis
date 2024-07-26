#import "template.typ": *

// ---------------------------------
// Global settings
// ---------------------------------
#set text(size: 13pt, spacing: 120%)
#set figure(gap: 0.5cm)
#show figure.caption: it => {
  align(left, it)
}
#show figure: set block(inset: (top: 0.5cm, bottom: 0.5cm))
#show math.equation: set block(above: 1.0cm, below: 1.0cm)
#show cite: set text(fill: blue)

    
// ---------------------------------
// Front matter
// ---------------------------------
#frontmatter(
  titlepage: titlepage(
    title: "Improving the usefulness of weather models output with machine learning",
    author: "Francesco Zanetta",
    birthdate: "01.04.1995",
    academictitle: "MSc Environmental Sciences",
    university: "ETH Zürich",
    supervisor: "Prof. Heini Wernli",
    coexaminers: (
      "Dr. Daniele Nerini", 
      "Dr. Mark Liniger",
      "Dr. Sebastian Lerch"
    )
  ),
)[



// Abstracts and acknowledgements
= Abstract
\ 
#set par(justify: true)
#lorem(500)
#pagebreak()

= Sommario
\ 
#lorem(500)
#pagebreak()

= Acknowledgments 
\
#lorem(200)
#pagebreak()
]



// ----------------------------------
// Main body
// ----------------------------------
#pagebreak(to: "even")

#show: mainbody 

#include "chapter-1/chapter-1.typ"

// pagebreak if needed 
#context {
  let count = counter(heading).get()
  if count != (1,) {
    pagebreak(to: "even")
  }
}

#include "chapter-2/chapter-2.typ"

// pagebreak if needed 
#context {
  let count = counter(heading).get()
  if count != (1,) {
    pagebreak(to: "even")
  }
}


#include "chapter-3/chapter-3.typ"

#pagebreak()

#include "chapter-4/chapter-4.typ"

#pagebreak(to: "even")

#include "chapter-5/chapter-5.typ"

// ---------------------------------
#pagebreak(to: "even") 
// Appendix
// ---------------------------------

#[
#show: appendix 
#include "appendix/appendix.typ"
]

// ---------------------------------
// Bibliography
// ---------------------------------

#show bibliography: set text(size: 11pt)

#bibliography("references.bib")