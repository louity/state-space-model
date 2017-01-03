var jsdom = require('jsdom');
jsdom.env(
  "<html><body></body></html>",        // CREATE DOM HOOK
  [ 'http://d3js.org/d3.v3.min.js'    // JS DEPENDENCIES online ...
  ],                 // ... & local-offline

  function (err, window) {


    var inputs = true

    var width = 300;
    var height = 200;

    var input_height = height / 4;
    var state_height = 2 * height / 4;
    var output_height = 3 * height / 4;

    var n_circle = 4
    var circle_spacing = width / (n_circle + 1)
    var circle_radius = 0.2 * circle_spacing

    var arrow_width = 0.5 * circle_radius
    var arrow_height = arrow_width

    var text_size = circle_radius

    var vertical_line_color = "black"
    var horizontal_line_color = "black"
    var curved_line_color = "black"

    var svg = window.d3.select("body")
      .append("svg")
      .attr("width", width)
      .attr("height", height)

    var defs = svg.append("defs")

    defs.append("marker")
      .attr("id", "arrow")
      .attr("markerWidth", arrow_width)
      .attr("markerHeight", arrow_height)
      .attr("refX", arrow_width)
      .attr("refY", "3")
      .attr("orient", "auto")
      .attr("markerUnits", "strokeWidth")
      .append("path")
        .attr("d", "M0,0 L0,6 L9,3 z")
        .attr("fill", "black")

    for (i=0; i < n_circle; ++i) {
      var index = i == n_circle - 1 ? 'T' : i+1
      if (i == n_circle - 2){
        var cx = (i+1) * circle_spacing
        var cy = state_height
        svg.append("text")
          .attr("x", cx)
          .attr("y", cy)
          .attr("font-size", 2 * text_size)
          .attr("text-anchor", "middle")
          .append("tspan").text("...")

          svg.append("line")
            .attr("y1", state_height)
            .attr("y2", state_height)
            .attr("x1", (i+1) * circle_spacing + circle_radius)
            .attr("x2", (i+2) * circle_spacing - circle_radius)
            .attr("stroke", horizontal_line_color)
            .attr("marker-end", "url(#arrow)")
      } else {
        // inputs
        if (inputs) {
          var ux = (i+1 + 0.5) * circle_spacing
          var uy = input_height
          svg.append("circle")
              .attr("cx", ux)
              .attr("cy", uy)
              .attr("r", circle_radius)
              .attr("stroke", "black")
              .attr("fill", "none")
          svg.append("text")
            .attr("x", ux)
            .attr("y", uy)
            .attr("font-size", text_size)
            .attr("text-anchor", "middle")
            .append("tspan").text("u")
              .append("tspan").attr("baseline-shift", "sub").text(index)
          }

        // states
        var xx = (i+1) * circle_spacing
        var xy = state_height
        svg.append("circle")
            .attr("cx", xx)
            .attr("cy", xy)
            .attr("r", circle_radius)
            .attr("stroke", "black")
            .attr("fill", "none")
        svg.append("text")
          .attr("x", xx)
          .attr("y", xy)
          .attr("font-size", text_size)
          .attr("text-anchor", "middle")
          .append("tspan").text("x")
            .append("tspan").attr("baseline-shift", "sub").text(index)

        // outputs
        var yx = (i+1) * circle_spacing
        var yy = output_height
        svg.append("circle")
            .attr("cx", yx)
            .attr("cy", yy)
            .attr("r", circle_radius)
            .attr("stroke", "black")
            .attr("fill", "none")
        svg.append("text")
          .attr("x", yx)
          .attr("y", yy)
          .attr("font-size", text_size)
          .attr("text-anchor", "middle")
          .append("tspan").text("y")
            .append("tspan").attr("baseline-shift", "sub").text(index)

        // vertical arrow
        svg.append("line")
          .attr("y1", state_height + circle_radius)
          .attr("y2", output_height - circle_radius)
          .attr("x1", (i+1) * circle_spacing)
          .attr("x2", (i+1) * circle_spacing)
          .attr("stroke", vertical_line_color)
          .attr("marker-end", "url(#arrow)")

        // horizontal arrow
        if (i < n_circle-1) {
          svg.append("line")
            .attr("y1", state_height)
            .attr("y2", state_height)
            .attr("x1", (i+1) * circle_spacing + circle_radius)
            .attr("x2", (i+2) * circle_spacing - circle_radius)
            .attr("stroke", horizontal_line_color)
            .attr("marker-end", "url(#arrow)")
        }
        //curved arrow
        if (inputs) {
          var x1 = ux
          var y1 = (uy + circle_radius)
          var x2 = (yx + circle_radius)
          var y2 = yy
          svg.append("path")
            .attr("d", "M " + x1 + " " + y1 + " C " + x1 +  " " + (y1 + 10)+ ", " + (x2 + 0.5 * circle_spacing) + " " + y2 +", " + x2 + " " + y2)
            .attr("stroke", "black")
            .attr("fill", "transparent")
            .attr("marker-end", "url(#arrow)")
        }
      }

    }
// END (D3JS) * * * * * * * * * * * * * * * * * * * * * * * *

  //PRINTING OUT SELECTION
    console.log('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="' + width + '" height="' + height + '">')
    console.log( window.d3.select("body").select('svg').html() );
    console.log('</svg>')
 } // end function
);
