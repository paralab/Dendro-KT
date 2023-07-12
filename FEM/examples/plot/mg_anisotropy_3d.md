
# Solver convergence on different meshes

```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
params:
  - name: structure
    select: {type: point, fields: [mesh_family]}
    bind: {input: select, options: [adapt, uniform]}
transform:
  - filter: {param: structure}
  - calculate: format(datum.scale_x, "d") + "/" + format(datum.scale_y, "d") + "/" + format(datum.scale_z, "d")
    as: scaling
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2
facet:
  row:
    field: scaling
    type: nominal
spec:
  mark: line
  encoding:
    x:
      field: vcycles
      type: quantitative
    y:
      field: rel_res_L2
      type: quantitative
      scale:
        type: log
    column:
      field: Cells (depth)
      type: nominal
      sort:
        field: cells
        order: descending
    color:
      field: solver
      type: nominal
      legend:
        orient: "top"

config:
  line:
    strokeWidth: 3
  header:
    titleAngle: 0
    labelAngle: 0
```


```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
transform:
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: format(datum.scale_x, "d") + "/" + format(datum.scale_y, "d") + "/" + format(datum.scale_z, "d")
    as: scaling

  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2

  - window:
    - op: first_value
      range: [-2,0]
      field: rel_res_L2
      param: 2
      as: lag2_res_L2
    - op: first_value
      range: [-1, 0]
      field: rel_res_L2
      param: 1
      as: lag1_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: min
        field: rel_res_L2
        as: min_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: max
        field: min_res_L2
        as: common_res_L2
    groupby: [solver]

  - filter: (datum.vcycles <= 5) || (datum.lag1_res_L2 >= datum.common_res_L2)

facet:
  column: {field: solver}
  row: {field: scaling}
spec:
  vconcat:
    - mark: line
      encoding:
        x:
          field: vcycles
          type: quantitative
        y:
          field: rel_res_L2
          type: quantitative
          scale:
            type: log
          axis:
            format: "0.0e"
        color:
          field: "Cells (depth)"
          type: nominal
          sort: {field: cells, order: ascending}
    - mark: bar
      encoding:
        x:
          field: vcycles
          type: quantitative
          aggregate: max
          axis:
            labels: false
            title: false
        y:
          field: "Cells (depth)"
          axis:
            title: false
          type: nominal
          sort:
            field: vcycles
            op: max
  resolve:
    scale:
      x: shared
resolve:
  axis:
    y: shared

config:
  line:
    strokeWidth: 3
  header:
    titleAngle: 0
    labelAngle: 0
```



## Same as above but trimmed down


```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
transform:
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: format(datum.scale_x, "d") + "/" + format(datum.scale_y, "d") + "/" + format(datum.scale_z, "d")
    as: scaling

  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2

  - window:
    - op: first_value
      range: [-2,0]
      field: rel_res_L2
      param: 2
      as: lag2_res_L2
    - op: first_value
      range: [-1, 0]
      field: rel_res_L2
      param: 1
      as: lag1_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: min
        field: rel_res_L2
        as: min_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: max
        field: min_res_L2
        as: common_res_L2
    groupby: [solver]

facet:
  column: {field: solver}
  row: {field: scaling}
spec:
  vconcat:
    - mark: line
      encoding:
        x:
          field: vcycles
          type: quantitative
        y:
          field: rel_res_L2
          type: quantitative
          scale:
            type: log
          axis:
            format: "0.0e"
        color:
          field: "Cells (depth)"
          type: nominal
          sort: {field: cells, order: ascending}
  resolve:
    scale:
      x: shared
resolve:
  axis:
    y: shared

config:
  line:
    strokeWidth: 3
  header:
    titleAngle: 0
    labelAngle: 0
```




-------------------------------------


# Solver convergence on different meshes

```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-nonuniform-cube.csv
params:
  - name: structure
    select: {type: point, fields: [mesh_family]}
    bind: {input: select, options: [adapt, uniform]}
transform:
  - filter: {param: structure}
  - calculate: format(datum.scale_x, "d") + "/" + format(datum.scale_y, "d") + "/" + format(datum.scale_z, "d")
    as: scaling
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2
facet:
  row:
    field: scaling
    type: nominal
spec:
  mark: line
  encoding:
    x:
      field: vcycles
      type: quantitative
    y:
      field: rel_res_L2
      type: quantitative
      scale:
        type: log
    column:
      field: Cells (depth)
      type: nominal
      sort:
        field: cells
        order: descending
    color:
      field: solver
      type: nominal
      legend:
        orient: "top"

config:
  line:
    strokeWidth: 3
  header:
    titleAngle: 0
    labelAngle: 0
```


```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-nonuniform-cube.csv
transform:
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)

  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2

  - window:
    - op: first_value
      range: [-2,0]
      field: rel_res_L2
      param: 2
      as: lag2_res_L2
    - op: first_value
      range: [-1, 0]
      field: rel_res_L2
      param: 1
      as: lag1_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: min
        field: rel_res_L2
        as: min_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: max
        field: min_res_L2
        as: common_res_L2
    groupby: [solver]

  - filter: (datum.vcycles <= 5) || (datum.lag1_res_L2 >= datum.common_res_L2)

facet:
  column: {field: solver}
  row: {field: scale_z}
spec:
  vconcat:
    - mark: line
      encoding:
        x:
          field: vcycles
          type: quantitative
        y:
          field: rel_res_L2
          type: quantitative
          scale:
            type: log
          axis:
            format: "0.0e"
        color:
          field: "Cells (depth)"
          type: nominal
          sort: {field: cells, order: ascending}
    - mark: bar
      encoding:
        x:
          field: vcycles
          type: quantitative
          aggregate: max
          axis:
            labels: false
            title: false
        y:
          field: "Cells (depth)"
          axis:
            title: false
          type: nominal
          sort:
            field: vcycles
            op: max
  resolve:
    scale:
      x: shared
resolve:
  axis:
    y: shared

config:
  line:
    strokeWidth: 3
```



```vis
data:
  url: Dendro-KT/savedata/2023-07-11-anisotropy-nonuniform-cube.csv
  
transform:
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toString("Scale=") + format(datum.scale_x, "d") + "/" + format(datum.scale_y, "d") + "/" + format(datum.scale_z, "d")
    as: scaling

  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver", "scaling"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2

  - window:
    - op: first_value
      range: [-2,0]
      field: rel_res_L2
      param: 2
      as: lag2_res_L2
    - op: first_value
      range: [-1, 0]
      field: rel_res_L2
      param: 1
      as: lag1_res_L2
    groupby: ["Cells (depth)", "solver", "scaling"]

  - joinaggregate:
      - op: min
        field: rel_res_L2
        as: min_res_L2
    groupby: ["Cells (depth)", "solver", "scaling"]

  - joinaggregate:
      - op: max
        field: min_res_L2
        as: common_res_L2
    groupby: [solver, "scaling"]

facet:
  column:
    field: solver
  row:
    field: scaling
    header:
      titleFontSize: 14
      titleAngle: 0
      labelAngle: 0
      title: null
spec:
  mark: line
  encoding:
    x:
      field: vcycles
      type: quantitative
    y:
      field: rel_res_L2
      type: quantitative
      scale:
        type: log
      axis:
        format: "0.0e"
    color:
      field: "Cells (depth)"
      type: nominal
      sort: {field: cells, order: descending}
      legend:
        orient: bottom
        columns: 3
        columnPadding: 20
resolve:
  axis:
    y: shared
  scale:
    x: shared

config:
  line:
    strokeWidth: 3
  headerColumn:
    titleFontSize: 14
```

