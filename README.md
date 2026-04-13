# Cancer Regression

Regression analysis predicting **cancer mortality rate** (`target_deathrate`) across US counties.

## Dataset

`cancer_reg.csv` — 3,047 US counties, 33 features aggregated from sources including the American Community Survey, clinicaltrials.gov, and cancer.gov (2009–2013).

**Target:** `target_deathrate` — mean per-capita (100k) cancer mortality rate per county.

## Feature Groups

| Group | Features |
|-------|----------|
| Cancer Burden | `avganncount`, `avgdeathsperyear`, `incidencerate` |
| Socioeconomic | `medincome`, `povertypercent`, `pctunemployed16_over`, `binnedinc` |
| Demographics | `popest2015`, `medianage`, `percentmarried`, `birthrate`, race % |
| Education | `pctnohs18_24`, `pcths25_over`, `pctbachdeg25_over`, ... |
| Healthcare | `pctprivatecoverage`, `pctpubliccoverage`, ... |
| Other | `studypercap`, `geography` |

## Missing Data

| Column | Missing % |
|--------|-----------|
| `pctsomecol18_24` | 75% |
| `pctprivatecoveragealone` | 20% |
| `pctemployed16_over` | 5% |

## Source

[Kaggle: Cancer Regression Dataset](https://www.kaggle.com/datasets/varunraskar/cancer-regression)
