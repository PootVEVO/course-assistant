# Team plan & GitHub workflow

This is the plan we used to split the work. Each item below becomes one GitHub **issue**. Every issue has
**completion checks**, the concrete things a reviewer verifies before closing it. Owners are left blank on
purpose: pick them as a team. `scripts/create_issues.sh` creates all of these issues for you.

| # | Issue | Completion checks | Owner |
|---|---|---|---|
| 1 | Repo, environment and secrets | Fresh clone + `pip install -r requirements-dev.txt` works on 2 laptops; `.env` is git-ignored; `git grep -n "<the real key>"` finds nothing | |
| 2 | Document preparation and the Materials tab | Add/remove works in the UI; re-adding a file is skipped; Week 2 PPTX converts and the slide preview matches the original deck; `tests/test_e2e.py::test_add_duplicate_replace_and_remove` and `::test_pptx_upload_is_converted_to_slide_images` pass | |
| 3 | Hybrid retrieval (BM25 + text + visual, RRF, reranker) | "Retrieval details" shows all three retrievers; the meme question puts Week 2 slide 33 in the top 4 | |
| 4 | Grounded answers (structured JSON, validation, support check) | Answers show doc + slide + excerpt + image; the unanswerable question says it isn't in the materials; no unverified citation is ever shown | |
| 5 | Quiz generation and scoring | Solutions stay hidden until Check/Show answer; score equals the answer key; every explanation's source supports the correct option (spot-check 10 questions) | |
| 6 | Gradio UI, light and dark mode | `tests/test_ui.py` passes; manual check of text, buttons and slide images in light **and** dark (`?__theme=dark`) | |
| 7 | Automated tests | `pytest` passes (unit + e2e + UI); `pytest -m live -v` passes on the class network | |
| 8 | Evaluation: reranker on vs off | `python -m eval.run_eval --materials course_files` run; `results/` committed; human-grading columns filled in `results/eval_results.csv` | |
| 9 | README report, SVG diagram, screenshots | A teammate follows README from a fresh clone without help; 2+ real screenshots in `docs/screenshots/`; diagram matches the code | |
| 10 | Final review and submission | All issues closed via merged PRs; professor has repo access; link submitted on Canvas and opened from a logged-out browser | |

## How we work in GitHub (short version)

1. **Issue**: a to-do item on GitHub with a description and completion checks. Assign yourself to one.
2. **Branch**: your own copy of the code to work in without breaking `main`:
   `git switch -c 3-hybrid-retrieval` (start the name with the issue number).
3. **Commit**: a saved snapshot with a message: `git add -A && git commit -m "Add RRF fusion (#3)"`.
4. **Push**: upload your branch with `git push -u origin 3-hybrid-retrieval`.
5. **Pull request (PR)**: on GitHub, "Compare & pull request". Write what changed and how you tested it, and put
   `Closes #3` in the description so the issue closes when the PR is merged.
6. **Review**: a teammate reads the diff, runs `pytest`, tries the feature, and leaves comments or approves.
   Everyone reviews at least one teammate's PR.
7. **Merge**: after approval, "Squash and merge". Then everyone runs `git switch main && git pull`.

Rules: never commit `.env`, keys, or course files (`course_files/` is git-ignored); run `pytest` before opening a PR.

## Glossary

- **RAG** (retrieval-augmented generation): find relevant course pages first, then have the LLM answer from them.
- **BM25**: classic keyword search scoring (rewards rare words that match the question).
- **Embedding**: a list of numbers representing meaning; similar texts/images have nearby embeddings.
- **RRF** (reciprocal rank fusion): merges several ranked lists by summing `1/(60 + rank)` for each item.
- **Reranker**: a model that looks at the question and each candidate page (text + image) together and scores relevance more carefully than the first-stage search.
- **Chunk**: a ~800-character piece of a page's text, tagged with its document and slide/page number.
