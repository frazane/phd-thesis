// ---------------------------------------------
// Front matter
// ---------------------------------------------

#let FONT_SIZE = 12.5pt

// A customized outline with a few adjustments compared to the default
#let customoutline = {
  show outline.entry.where(
      level: 1
    ): it => {
      set text(weight: "bold")
      if it.at("label", default: none) == <modified-entry> {
        it // prevent infinite recursion
      } else {

        let vgap = 1pt
        if it.body == [Abstract]{
          vgap = -2pt
        }
        if it.body == [Sommario]{
          vgap = -2pt
        }
        if it.body == [Acknowledgments]{
          vgap = -2pt
        }
        if it.body == [Appendix] {
          [ \
            #outline.entry(
              it.level,
              it.element,
              [#v(vgap) #it.body],
              "",
              ""
            ) <modified-entry>
          ]
        } else {
          [
            #outline.entry(
              it.level,
              it.element,
              [#v(vgap) #it.body],
              "",
              it.page
            ) <modified-entry>
          ]
        }
      }
    }
  show outline: set text(size: 12pt)
  outline(
    indent: 4%,
    title: [#text(size: 20pt)[Contents #v(20pt)]], 
    depth: 3
  )
}


// Title page for ETH Zürich doctoral dissertations
#let titlepage(
  title: str,
  author: str,
  birthdate: str,
  academictitle: str,
  supervisor: str,
  coexaminers: (),
  university: "ETH Zürich",
  dissnumber: "00000",
) = {
  set text(size: 15pt)
  set align(center)
  
  // dissertation number
  [DISS. ETH NO. #dissnumber]
  v(2.5cm)
  
  // title
  upper[_*#title*_]
  v(2.5cm)

  set text(size: FONT_SIZE)
  // body
  [
    A thesis submitted to attain the degree of \ \ 
    DOCTOR OF SCIENCES \ 
    (Dr. sc. ETH Zürich) \ \ \
    presented by \ \ \
    _#upper[#author]_ \ \
    _ #academictitle, #university _ \ \ \
    born on _ #birthdate _ \ \ \ \ \
    accepted on the recommendation of \
    _ #supervisor _
    #for name in coexaminers {
      [\ _ #name _]
    }
  ]
  
  v(2cm)
  
  str(datetime.today().year())
}


// Front matter function
#let frontmatter(
  body,
  titlepage: none,
) = {
  set page(
    margin: (inside: 2.5cm, outside: 2cm, y: 2.5cm),
    number-align: center,
    numbering: "1",
  )


  titlepage

  // font size of heading level 1
  show heading.where(
    level: 1
  ): it => [
    #set text(size: 22pt)
    #it.body \
  ]

  // roman numbering for front matter
  set page(numbering: "i")

  // front matter body
  body

  // pretty outline
  pagebreak(to: "even")
  customoutline
}

// ---------------------------------------------
// Main body
// ---------------------------------------------

#let mainbody(body) = {
  set page(
    margin: (inside: 2.9cm, outside: 2.4cm, y: 2.7cm),
    number-align: center,
    numbering: "1",
  )
  // set page(margin: auto)
  set bibliography(style: "american-psychological-association")
  counter(page).update(1)
  set list(indent: 20pt)
  set enum(indent: 20pt)
  set text(size: FONT_SIZE)
  set par(justify: true)
  set par(leading: 0.7em)

  // headings
  set heading(numbering: "1.1")

  // heading level 1
  show heading.where(
    level: 1
  ): it => [

    // #counter(figure).update(0)
    #counter(math.equation).update(0)
    
    #set heading(supplement: [Chapter])
    
    // actual heading text
    #v(5cm)
    #set text(size: 25pt)
    Chapter #counter(heading).display() \ \ 
    #set text(size: 23pt)
    #counter(math.equation).update(0)
    #counter(figure.where(kind: image)).update(0)
    #counter(figure.where(kind: table)).update(0)
    #counter(figure.where(kind: raw)).update(0)
    #it.body
    \ \
  ]
  
  // set figure(numbering: it => {
    // [#counter(heading).display()]
  // })
  
  // heading level 4
  show heading.where(
    level: 2
  ): it => [
    #v(0.5cm)
    #counter(heading).display() #it.body
    #v(0.3cm)
  ]
  
  // heading level 4
  show heading.where(
    level: 3
  ): it => [
    #v(0.7cm)
    #counter(heading).display() #it.body
    #v(0.5cm)
  ]
  
  // heading level 4
  show heading.where(
    level: 4
  ): it => [
      #parbreak()
      #v(0.4cm)
      #text(style: "italic", weight: "regular", it.body + ".")
      #v(0.2cm)
  ]

  // reference to heading

  // equations numbering based on section
  set math.equation(numbering: num =>
    "(" + (counter(heading.where(level: 1)).get() + (num,)).map(str).join(".") + ")")
  
  set figure(numbering: num =>
    numbering("1.1", counter(heading).get().first(), num)
  )
  // set figure(numbering: num => {
  //   locate(loc => {
  //     let chap_num = counter(heading.where(level: 1)).at(loc).first()
  //     let chap_loc = query(heading.where(level: 1).before(loc)).last().location()
  //     let fig_offset = counter(figure).at(chap_loc).first()
  //     str(chap_num) + "." + str(num - fig_offset)
  //   })
// })    
  // show the body :-) 
  body
}

// ---------------------------------------------
// Appendix
// ---------------------------------------------

#let appendix(body) = {
  set page(
    margin: (inside: 2.5cm, outside: 2cm, y: 1.75cm),
    number-align: center,
    numbering: "1",
  )
  // set page(margin: auto)
  set text(size: FONT_SIZE)
  set par(justify: true)
  show heading.where(
    level: 1
  ): it => [
    #v(5cm)
    #set text(size: 23pt)
    #it.body
    \ \
  ]
  
  heading("Appendix", numbering: none)
  
  counter(heading).update(0)
  
  show heading.where(
    level: 1
  ): it => [
    #pagebreak()
    #v(5cm)
    #set text(size: 23pt)
    #counter(heading).display()
    #it.body
    \ \
  ]
  set heading(numbering: "A.1")
  body
}




// 
// #titlepage(
//   title: "Exploring the Paradox of Cat Breadth Theory: A Quantum Mechanical Approach to Feline Rectangularity",
//   author: "Felicity Furrington",
//   birthdate: "08.08.1997",
//   academictitle: "MSc Feline Physics",
//   university: "ETH Zürich",
//   supervisor: "Prof. Fluffy McPurrson",
//   coexaminers: (
//     "Dr. Whisker Wilde", 
//     "Dr. Leonardo Pawsley",
//     "Prof. Clawdia Scratchwell"
//   )
// )








// #titlepage(
//   title: "Exploring the Paradox of Cat Breadth Theory: A Quantum Mechanical Approach to Feline Rectangularity",
//   author: "Felicity Furrington",
//   birthdate: "08.08.1997",
//   academictitle: "MSc Feline Physics",
//   university: "ETH Zürich",
//   supervisor: "Prof. Fluffy McPurrson",
//   coexaminers: (
//     "Dr. Whisker Wilde", 
//     "Dr. Leonardo Pawsley",
//     "Prof. Clawdia Scratchwell"
//   )
// )