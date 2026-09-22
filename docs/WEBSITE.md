# Project website

The recruiter-facing website lives in `website/`. It uses plain HTML, CSS, and a small
JavaScript enhancement for copying installation commands. No package installation,
application server, simulation worker, database, or runtime build is required to serve it.

## Local preview

From the repository root:

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:4173`. Stop the local server with Ctrl+C when finished. This preview
server is unrelated to optimization jobs or the hosted site.

## Vercel configuration

Use the existing `aster-hpo` project with these settings:

- Root directory: `website`
- Framework preset: Other (`null` in the API)
- Build command: empty
- Install command: empty
- Output directory: `.`

The configuration file is `website/vercel.json`. The project root must be `website`:
publishing `.` from the repository root would include unintended files and can trigger
Python application detection. The Python library is not an ASGI/WSGI application and
does not need a Vercel Python entrypoint. Keep the production branch set to `main`.

The hosted page serves saved files. It never launches experiments in response to a visit.

## Evidence snapshot

`website/evidence/manifest.json` identifies the library revision, command, seed, capture
date, and hashes for the raw JSON in `website/evidence/smoke/`. The capture uses the
existing noisy-quadratic example, not a financial simulation or a performance benchmark.
The site's figures show per-candidate allocation for that specific saved run.

To produce a new snapshot, first install the package as described in the root README:

```bash
python examples/noisy_quadratic.py --workers 2 --configs 16 --seed 17 --output website/evidence/smoke
```

Then update the manifest, its file hashes, the displayed capture date, counts, and figure
descriptions to match the new output. Run `python scripts/render_website_evidence.py`
to regenerate the static SVG allocation figures (16 candidates, replication cap 27). Do not change the snapshot during deployment. Asynchronous
scheduling can change allocation on later runs even with identical seeds.

Keep planned capabilities and completed work distinct. Do not present this smoke test as
a speedup, quality guarantee, financial result, or large-scale execution benchmark.

## Verification before publishing

Check local links and evidence files; run `node --check website/assets/site.js`; inspect
the page at desktop and mobile widths, with enlarged text and JavaScript disabled; and
check keyboard navigation and the copy control. When clipboard access is unavailable,
the control selects the commands for manual copying.

After deployment, verify that the production URL is publicly readable without signing in
and that CSS, JavaScript, and evidence JSON load successfully. Preview deployments may
remain protected. Confirm that the deployment contains no Python functions.

## Visual direction

The site uses a compact scientific-software layout: persistent section navigation,
an API example, allocation figures drawn from the saved records, and research notes.
Optuna and Dask informed the emphasis on runnable examples and execution data.
All fonts are system fonts; no external assets or frontend dependencies are required.
