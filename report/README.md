# report/

- `main.tex` is the report. Compile it with `pdflatex main.tex` (run it three times for the table of contents and the references) or upload the folder to Overleaf.
- `figures/` and `tables/` hold the images. **The tex file loads them by name** (`figures/<name>.png`, `tables/<name>.png`), and the names are exactly the names the notebook saves on Drive in `report_assets/figures` and `report_assets/tables`. Copy the files from Drive into these two folders (or unzip `report_assets.zip` from Block 17b here). A missing image shows as a grey box, so the file always compiles.
- `logo.png` (university logo, optional) goes next to `main.tex`.
- The four diagrams (`fig_workflow`, `fig_architecture`, `fig_agent_loop`, `fig_mlp_architecture`) are already in `figures/`; the four screenshots of the application (`fig_app_ask`, `fig_app_trace`, `fig_app_predictor`, `fig_app_about`) have to be taken from the running application.
- Before submission: set `\showtodofalse` near the top of `main.tex` (this hides the grey notes) and fill the macros `\guidename`, `\guidedesig`, `\repourl` and `\appurl`.
