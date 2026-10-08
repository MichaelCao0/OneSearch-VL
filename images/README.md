# README figure sources

The paper figures are exported from the manuscript **OneSearch-VL: Unified
Multimodal Deep Research Agent for Image and Video**. Their contents are
unchanged; the root README captions follow the corresponding paper captions.

| Asset | Manuscript source | Paper figure label |
|---|---|---|
| `data_pipeline.png` | `figures/pipeline-data.pdf` | `fig:data-pipeline` |
| `evgr.png` | `figures/pipeline-reward.pdf` | `fig:evgr` |
| `benchmarks.png` | `figures/bench.pdf` | `fig:benchmark-overview` |
| `case_mi_arithmetic.png` | `figures/cases/mi_arithmetic.pdf` | `fig:case-mi-arithmetic` |

Each single-page PDF was rendered with Poppler, preserving its aspect ratio:

```bash
pdftoppm -f 1 -singlefile -scale-to 2400 -png SOURCE.pdf OUTPUT_STEM
```

`logo.png` is the supplied `onesearchvl-typeset-v14.png`, copied without changes.
