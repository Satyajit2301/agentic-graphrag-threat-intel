# report/

- `main.tex` is the report. Compile it with `pdflatex main.tex` (run it three times for the table of contents and the references) or upload the folder to Overleaf.
- `figures/` and `tables/` hold the images. **The tex file loads them by name** (`figures/<name>.png`, `tables/<name>.png`), and the names are exactly the names the notebook saves on Drive in `report_assets/figures` and `report_assets/tables`. Copy the files from Drive into these two folders (or unzip `report_assets.zip` from Block 17b here). A missing image shows as a grey box, so the file always compiles.
- `logo.png` (university logo, optional) goes next to `main.tex`.
- The four diagrams (`fig_workflow`, `fig_architecture`, `fig_agent_loop`, `fig_mlp_architecture`) are already in `figures/`; the six screenshots of the application (`fig_app_ask`, `fig_app_trace`, `fig_app_graph`, `fig_app_predictor`, `fig_app_predictor_2`, `fig_app_about`) are the ones taken from the running application.
- All names, links and the guide are already set in the macros at the top of `main.tex`; `\showtodofalse` hides the grey notes. If you have the university logo, save it as `logo.png` next to `main.tex`; it then appears on the title page.
