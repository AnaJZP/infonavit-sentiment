# Raw Data

The raw data consists of six Excel files provided by INFONAVIT's
communications monitoring department:

| File | Period | Rows (approx.) |
|------|--------|----------------|
| `BD 2019.xlsx` | Nov 2019 – Dec 2019 | ~8,000 |
| `BD 2020.xlsx` | Jan 2020 – Dec 2020 | ~16,000 |
| `BD 2021.xlsx` | Jan 2021 – Dec 2021 | ~23,000 |
| `BD 2022.xlsx` | Jan 2022 – Dec 2022 | ~28,000 |
| `BD 2023.xlsx` | Jan 2023 – Dec 2023 | ~37,000 |
| `BDD 2024.xlsx` | Jan 2024 – Jul 2024 | ~48,000 |

## Columns

Each file contains the following core columns:

| Column | Description |
|--------|-------------|
| `Fecha` | Publication date |
| `Medio` | Media outlet name |
| `Autor` | Author / journalist |
| `Tipo` | Media type (Prensa, Portal, Radio, TV, Revista) |
| `Cobertura` | Geographic coverage (Nacional, Estatal, Local, Internacional) |
| `Estado_Origen` | Mexican state of origin |
| `Encabezado` | Headline text (main input for NLP) |
| `Actor` | Mentioned institutional actor |
| `Mencion` | Type of mention |
| `Tema` | Manually assigned topic |
| `Tono` | Manual sentiment label (Positivo / Negativo / Neutral) |
| `Origen` | Source type |
| `Tipo_de_publicacion` | Publication subtype |
| `Equivalencia_publicitaria` | Advertising equivalency value (MXN) |

Some files also include: `Pais`, `Municipio`, `Alcance`, `Testigo`.

## Access

These files are not included in the repository due to size and
licensing constraints. To reproduce the analysis from scratch,
place the six Excel files in this directory and run the pipeline
starting from `scripts/01_consolidate_data.py`.

Alternatively, use the pre-computed results in `results/` to
verify the downstream computations (RRS model, visualizations).
