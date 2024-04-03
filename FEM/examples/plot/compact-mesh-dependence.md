```vis
data:
#######  url: code/paralab/Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
  url: code/paralab/Dendro-KT/savedata/2023-07-11-anisotropy-nonuniform-cube.csv
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

height: 80

encoding:
  x:
    type: quantitative
    scale:
      type: log
      base: 2
      domainMin: 2048
      domainMax: 131072
    axis:
      values: [2048, 8192, 32768, 131072]
    title: Cells

  y:
    type: quantitative
    scale:
      type: linear
      zero: true
    title: ["V-cycles", "per 10", "digits"]

layer:
  - mark: line
    encoding:
      detail:
        field: stretch
        type: nominal
      x:
        field: cells
      y:
        field: vcycles_per_10_digits
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

  - mark:
      type: text
      align: left
      dx: 10
    encoding:
      x:
        field: cells
        aggregate: max
      y:
        field: vcycles_per_10_digits
        aggregate: {argmax: vcycles_per_10_digits}
      text:
        field: stretch


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


