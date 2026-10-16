#!/usr/bin/env bash
# Creates the team's GitHub issues (see docs/TEAM_WORKFLOW.md).
# Needs the GitHub CLI: https://cli.github.com  ->  gh auth login
# Run from inside the cloned repo:   bash scripts/create_issues.sh
set -euo pipefail

mk() { gh issue create --title "$1" --body "$2"; }

mk "Repo, environment and secrets" "- [ ] Fresh clone + \`pip install -r requirements-dev.txt\` works on two laptops
- [ ] \`.env\` created from \`.env.example\` and git-ignored
- [ ] \`git grep\` for the real key finds nothing"

mk "Document preparation and Materials tab" "- [ ] Add and remove files from the UI
- [ ] Re-adding the same file is skipped (no duplicates)
- [ ] Week 2 PPTX converts with LibreOffice; slide preview matches the original deck
- [ ] Related e2e tests pass"

mk "Hybrid retrieval: BM25 + text + visual, RRF, reranker" "- [ ] Retrieval details show keyword, text and visual ranks
- [ ] Meme question retrieves Week 2 slide 33 in the top 4
- [ ] Reranker can be switched on/off"

mk "Grounded answers: structured JSON, validation, support check" "- [ ] Answer shows document, slide/page, excerpt and slide image
- [ ] Unanswerable question is acknowledged as missing
- [ ] Unverifiable citations are never shown"

mk "Quiz generation and scoring" "- [ ] Solutions hidden until Check / Show answer
- [ ] Score matches the answer key
- [ ] Spot-check 10 questions: answerable from the material and explanation sources support the answer"

mk "Gradio UI: light and dark mode" "- [ ] tests/test_ui.py passes
- [ ] Manual check of text, controls and slide images in light and dark (?__theme=dark)"

mk "Automated tests" "- [ ] \`pytest\` passes
- [ ] \`pytest -m live -v\` passes on the class network"

mk "Evaluation: reranker on vs off" "- [ ] \`python -m eval.run_eval --materials course_files\` run
- [ ] results/ committed; human-grading columns filled in results/eval_results.csv
- [ ] Interpretation written in README"

mk "README report, SVG diagram, screenshots" "- [ ] Teammate follows README from a fresh clone without help
- [ ] At least 2 real screenshots in docs/screenshots/
- [ ] docs/architecture.svg matches the final app"

mk "Final review and submission" "- [ ] All issues closed via merged PRs
- [ ] Professor has repo access
- [ ] Link submitted on Canvas and checked"
