```vis
data:
  url: code/paralab/Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
transform:
  - filter: indexof(datum.solver, "accelerated") != -1
  - calculate: replace(datum.solver, "accelerated_", "cg+")
    as: solver
  - calculate: toString("Anisotropy(") + format(datum.scale_x, "d") + ":" + format(datum.scale_y, "d") + ":" + format(datum.scale_z, "d") + ")"
    as: stretch
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toNumber(datum.cells)
    as: cells
  - calculate: toNumber(datum.vcycles)
    as: vcycles
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver", "stretch"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2
  - joinaggregate:
    - op: max
      field: vcycles
      as: max_vcycles
    groupby: ["Cells (depth)", "solver", "stretch"]
  - filter: "datum.vcycles == datum.max_vcycles"
  - calculate: -log(datum.rel_res_L2)/LN10 / datum.vcycles
    as: digits_per_vcycle
  - calculate: datum.vcycles * 10.0 / (-log(datum.rel_res_L2)/LN10)
    as: vcycles_per_10_digits

facet:
  row:
    field: stretch
    header:
      title: false
      labelOrient: top
      labelPadding: -20
      labelAlign: left
      labelAnchor: start
spacing: 5

spec:
  height: 60
  mark: line
  encoding:
    detail:
      field: stretch
      type: nominal
    x:
      field: cells
      type: quantitative
      scale:
        type: log
        base: 2
      axis:
        values: [4096, 32768, 262144]
      title: Cells
    y:
      field: vcycles_per_10_digits
      type: quantitative
      scale:
        type: linear
        zero: true
      title: ["V-cycles", "per 10", "digits"]
      sort:
        field: cells
    color:
      field: solver
      type: nominal
      legend: null
      scale:
        scheme: set1
    shape:
      field: solver
      type: nominal
      legend:
        title: null
        orient: none
        direction: horizontal
        legendX: -80
        legendY: -30
        columnPadding: 15

config:
  line:
    strokeWidth: 3
    point: true
  point:
    size: 75
  text:
    font: Times
  style:
    guide-label:
      font: Times
      fontSize: 12
    guide-title:
      font: Times
      fontSize: 14
      fontWeight: normal
    group-title:
      font: Times
  axisY:
    titleAngle: 0
```


