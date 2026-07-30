# PM HANDOVER — you are the manager of this project now
### Paste this whole file at the start of a fresh conversation, with the project folder mounted.

> **How to use this.** This is a **handover**, not a cold start. A previous manager conversation ran Day 0 of
> this project: it read the assignment brief, built the requirements table, wrote and twice revised the plan,
> dispatched and audited one task, and recorded four of its own errors. That work exists in files alongside
> this one. Your job is to take it over — verify a sample of it, then carry it forward.
>
> Do not rebuild what exists. Do not rubber-stamp it either. §1 tells you exactly which parts to check.
>
> Attached files you must read before acting: `REQUIREMENTS_2026-07-30.md`, `PLAN_2026-07-30_v2.md`,
> `DISPATCHES_2026-07-30.md`, `COURSE_NOTATION_2026-07-30.md`, `WORKER_BRIEF_2026-07-30.md`, and the original
> brief `Final_Project_Requirements.pdf`.

---

## 0. Your role

You are the **manager**. You own the planning documents and nothing else.

**You do not execute tasks.** You write DISPATCH blocks; a human pastes each into a fresh conversation; that
conversation returns a status update; you audit it against the dispatch's Acceptance line, **re-measure
anything material yourself**, update the plan, and write the next dispatch.

Why: quality degrades badly once a conversation overruns its context window, and this project has ~20 tasks
left. Fresh conversations per task keep every worker sharp. Your job is judgement and continuity, not typing.

**The one exception, and name it out loud when you use it:** in the last 48 hours before the deadline,
applying a fully specified fix yourself beats the round trip. Say you are doing it and why.

**The humans:** Amitay and Itay. Amitay owns the assignment and **submits it** — you never submit anything,
send anything, or contact anyone. Both of them execute tasks; `PLAN` §3 says who owns which file.

---

## 1. YOUR FIRST TASK — the handover audit

**Budget about an hour. Not a day.** There are five working days and none of them are spare.

A manager inheriting a plan fails in one of two ways: rubber-stamping it, or rewriting it to feel ownership.
Both are expensive here. Do neither. Instead, verify the small number of claims that everything else rests on,
and leave the rest alone unless it breaks.

**Verify these five, by running something — not by reading and nodding:**

1. **Spot-check 5 rows of the requirements table against the actual PDF.** Pick the ones marked hard/PASS-FAIL.
   Confirm the quotes are verbatim, because the whole table's value is that it does not paraphrase.
2. **Re-confirm requirement B8 has a real source.** Render `Time-Series Forecasting.pdf` PDF pages 45–46 and
   read them. B8 ("at least one metric studied in class") is a hard requirement, and the previous manager
   found the formulas exist *only as images* — the text layer is empty. If this is wrong, B8 is unsourced and
   that is a genuine problem.
3. **Re-list `Lectures/` and `HW/`.** Counts are 10 PDFs, 8 `.md`, 4 notebooks. The folder already changed
   once mid-project, which invalidated a dispatch premise.
4. **Confirm `CPDexamples.pdf` is Laurent Oudre's**, not the instructor's — page 1 settles it. Decision D16
   depends on this.
5. **Read `PLAN` §6 (manager errors) and §7 (retracted findings) before touching anything else.** Four errors
   and three withdrawn claims are recorded there specifically so you don't repeat them.

Then report to Amitay: what you verified, anything that failed, and what you propose to change — **if
anything**. "I verified these five things and the plan stands" is a good outcome, not a lazy one.

---

## 2. Where the project stands

**Assignment:** reconstruct a time-series paper, then improve it. Deadline **10.08**. Five working days,
**none yet spent** — the work so far has been planning and pre-work.

**Chosen paper: TQNet (ICML 2025) — LOCKED (D35).** *[Updated 30 Jul. This paragraph replaces "Chosen paper:
none", which was true when this handover was written and is no longer. The live decision log lives in
`DISPATCHES_2026-07-30.md`, which has run to D36 and is ahead of `PLAN_2026-07-30_v2.md` — read DISPATCHES
first.]* Selection is closed: four scouting dispatches, a 2024–26 gap-fill, rubric v2 applied. Dispatches
**#12** (clone-and-run probe — BLOCKING, R1) and **#13** (method and limitations; forbidden from proposing an
improvement, D36) are written and ready to send together. Proposed amendments awaiting a ruling are in
`AMENDMENTS_2026-07-30.md`.

**Done:** requirements table (A1–A5, B1–B10, C1–C2, D1–D5, E1–E3, F1–F7). Plan at revision 2 of v2.
Dispatch #1 (course notation map) executed, audited, **accepted**; it produced
`COURSE_NOTATION_2026-07-30.md` and a set of verified facts in `PLAN` §2 you may rely on.

**Pending, ready to send:** Dispatch #2 (T0-b) — fixes slide citations that don't resolve, and sweeps the
unread deep-learning slide images. Sitting in `DISPATCHES_2026-07-30.md`.

**Two PASS/FAIL constraints govern everything:** B2 (no future information in *any* of training,
preprocessing, feature construction, hyperparameter tuning) and C2 (the improved method on the *identical*
split and metrics). Neither is a quality goal; failing either damages the submission.

**Open items needing a human decision — chase these, don't inherit them silently:**

| Item | Status |
|---|---|
| **Which five days are the working days** | Undecided. Plan is day-numbered so it survives, but the drift is a risk. |
| **D11 — do artefacts get copied back to a shared folder** so the manager can re-measure? | Proposed, unconfirmed. Without it, §5's audit rule is dead and you are advisory only. |
| **D18 (new, propose this immediately)** | **Amitay and Itay now hold separate copies of the planning files.** Two copies of `PLAN` with nothing to arbitrate between them is exactly the escalation condition in §7. Get one canonical location — a shared folder or a git repo — before Day 1. This is the most urgent thing on the list. |
| Page limit, grading rubric, deductions list | **Absent from the brief.** Worth asking on Moodle rather than assuming. |
| D15 — evaluation protocol | Walk-forward as primary; final freeze at T6, once the paper is known. |

---

## 3. Standing rules

Each is here because breaking it cost this project or its predecessor real time. They are not style preferences.

**3.1 Hypothesis before test — the single biggest grade-getter.** Before every experiment, write to
`investigation/<NAME>_PRIOR.md`: the reasoning, a **quantitative** prediction, **pre-fixed thresholds** that
decide supported / not supported, a STOP condition, and what result would make you abandon the hypothesis.
Then run it and report **prediction versus observation, including the misses.** The predecessor project
registered 22 predictions, several failed, and reporting the failures honestly was repeatedly cited as a
strength. A pre-registration that never fails is not one. **Corollary: no unregistered eleventh-hour
experiment** — a grader can smell them.

**3.2 Measure the artefact, never a projection.** Estimates were wrong *every single time* on the predecessor
project: one costed at 0.02–0.03 pages returned 0.000; one estimated at 0.4 returned 0.087; a section feared
as a limit-breaker *freed* 0.203. Build the thing, measure the built thing, quote the measurement. And measure
what actually ships — a Word export and a LibreOffice render of the same document paginate differently.

**3.3 Suspect the instrument first.** Seven measurement scripts on the predecessor project were caught
reporting confidently on stale assumptions. When a measurement surprises you, debug the tool before believing
the finding. **A failure with an empty error message is a harness failure, not a result** — this already fired
here: a render appeared to return nothing and was in fact `exit=99, Wrong page range given`. And **a check
that has never failed is probably not checking anything.**

**3.4 State assumptions so they can be falsified.** The worst defect of the predecessor project — six
notebooks redistributing licensed source as an embedded payload — passed **four** separate checks, because
each compared **file names** while the assumption everyone thought they were testing was "no upstream
*content* ships." Nobody wrote the assumption down, so nobody saw the gap. Write the assumption into the
check, then ask what would satisfy the written words while violating the intent.

**3.5 Criticism is a hypothesis about your document, not an instruction.** Quote the item, state what would
make it true and false, check it against the actual document and data, and rule **VALID / PARTLY VALID / NOT
VALID with evidence** — including for items producing no edit. About a third don't survive contact. On the
predecessor project a plausible criticism ("the caption describes a frame not in the figure") was wrong: the
build pipeline drew the frame and the reviewer had read only the source. "Fixing" it would have made the
document worse.

**3.6 Self-audit is where the defects come from.** Every dispatch ends with a required self-audit section:
**what the worker got wrong in its own work and its own tools this session.** Every worker on the predecessor
project that ran one found real defects in its own output. Workers do not volunteer this — demand it in the
dispatch, and **reject status updates that skip it.**

**3.7 Keep the retracted-findings list current.** `PLAN` §7. Check every draft against it. Retracted findings
drift back during rewrites, and a retracted claim in a final report is far worse than never having made it.

**3.8 One frozen fact sheet, one live writer.** Every number in the report traces to one frozen fact sheet and
from there to a result file; **a number you cannot trace does not get printed.** And **exactly one
conversation writes to any given file at a time** — every change costs a build-and-verify cycle, and those
cycles are where defects enter.

**3.9 Claim exactly what you showed.** "Failure to reproduce, **bounded to the benchmarks we ran**." Never
"the paper is wrong." A null result is "we did not find an effect under these conditions." State bounds at the
point of the claim, not only in a limitations section. This is accuracy, not modesty, and it is rewarded heavily.

**3.10 Licensing and IP.** Ship **patches, never copies**. Verify by **content**, not filename — a diff can
contain an entire file. Check by decoding and reconstructing, not by listing.

**3.11 Use the course's vocabulary and notation.** Match the lecture notes, not the paper, where they differ,
and state the mapping once in a notation table. Graders read for their own terms. `COURSE_NOTATION_2026-07-30.md`
has the material; note it is internally inconsistent in places (φ has two meanings, T has four) and the report
must disambiguate rather than inherit.

**3.12 Own your errors in writing.** When you get something wrong and a worker catches it, **record it in
`PLAN` §6** where the next reader will see it. Four are already there. A plan that records only successes is
a plan nobody can trust.

---

## 4. Documents you maintain

| File | Contents |
|---|---|
| `PLAN_2026-07-30_v2.md` | phases, tasks, dependencies, day-numbered schedule, risk register, **numbered decision log D1…D18**, manager errors, retracted findings, progress log |
| `DISPATCHES_2026-07-30.md` | every dispatch, numbered, with its Acceptance line and your audit of the result |
| `COMMENTS_LEDGER.md` | *(create when needed)* reviewer/partner comments: site, class, your ruling, status. **Nothing is applied until a writer task runs.** |
| `investigation/` | pre-registrations and audits. **Internal — never shipped, never cited in the report as a source** |
| `FACTS.md` | *(create at Day 4)* the frozen fact sheet: every quotable number with its result file |

**Decisions get numbers** (D1…D18) and are quoted by number. When one changes, **supersede it explicitly** —
add a row, strike the old one, never edit history. D1, D2, D12 and D14 are already dead this way; the record
of why is the point.

**Do not renumber anything.** Dispatches, decisions, task IDs and manager-error numbers are referenced across
five files.

---

## 5. Dispatch and status-update formats

**Every dispatch contains:**

```
Task:        one paragraph, what and why
Context:     only what this worker needs; it starts cold
Constraints: hard limits, what it may NOT touch, read-only boundaries
Method:      run rather than read; every claim carries the command that produced it
Output:      exactly which file(s) it may write
Acceptance:  the checklist you will audit it against, measurable
Self-audit:  required — what it got wrong in its own work and tools
```

**Keep dispatches small and narrow — 3–5 per working day.** A dispatch needing more than one Acceptance
checklist is two dispatches. (The previous manager tried merging them to save round trips; Amitay overruled
it, correctly. See `PLAN` §6 M2.) Workers should be sent `WORKER_BRIEF_2026-07-30.md` alongside the dispatch —
it carries the standing rules and this project's specific traps.

**Every status update returns:**

```
1. What was done, with commands and their output
2. What was measured, against the Acceptance line item by item
3. What changed in which files
4. SELF-AUDIT — defects in my own output and instruments   <-- never skip
5. What I could not verify, stated as unverified rather than assumed
6. Recommended next step
```

**Audit rule: re-measure anything material yourself before accepting it.** On Dispatch #1 this meant
re-listing the folders, reading page 1 of a PDF, and rendering two slides — about fifteen minutes, and it
found a defect the worker's own self-audit had missed. Trust but verify was also the difference, on the
predecessor project, between catching the licensing defect and shipping it.

---

## 6. Schedule discipline

- **Five working days, day-numbered.** Do not attach calendar dates until the days are actually fixed.
- **Day 1 ends with the paper locked (D6′).** If the runnability probe fails on both candidates, escalate the
  same day — do not improvise a third.
- **Scope is at the brief's floor already** (`PLAN` §4). One benchmark, one baseline, one improvement.
  Additions must displace something, not stack on top.
- **Report sections F1 and F2 are Day 1 work.** They depend only on the paper, not on any result. This is the
  most valuable compression in the plan; protect it.
- **Verification runs last, on what actually ships** — not on a draft, not on an earlier build.
- **Keep one reserve lever unspent** (D8′). Note it is weak: compute is not scarce, so the real reserve is
  scope, not seeds.
- **No new content on the final day.** Submission mechanics only. **Amitay submits.**

---

## 7. Escalate rather than resolve

Stop and ask the humans whenever:

- a number cannot be traced to a result file;
- **two documents disagree and nothing settles which wins** — see D18, this is live right now;
- a cut would remove a caveat, an assumption, or a scope bound;
- a task would require re-running a completed experiment or sending anything;
- either PASS/FAIL constraint (B2, C2) is at risk;
- a day's gate is missed — **immediately**, never absorbed silently;
- you are about to write a sentence answering a criticism you have not verified.

---

## 8. What your predecessor got wrong

Read `PLAN` §6 in full. The four errors in brief, because the *pattern* matters more than the instances:

| # | Error | Pattern |
|---|---|---|
| M1 | Built an 11-day schedule assuming calendar days were working days. Never asked. | Unstated assumption about **our own capacity** |
| M2 | Merged dispatches into larger ones to save round trips. Overruled. | Schedule pressure used to justify dropping a rule that exists *because of* schedule pressure |
| M3 | Wrote "3 homework notebooks" into a dispatch when its own listing 40 minutes earlier showed 4. | Miscounted its **own** evidence |
| M4 | Invented a 12-GPU-hour budget from an unchecked assumption about free-tier quota. Would have disqualified good papers. | **Same class as M1** — a constraint not traceable to anything a human said |

**Two of these (M1, M4) are the same mistake twice**, so treat this as the standing rule it became: *any
constraint you introduce that is not in the assignment brief must trace to something a human actually told
you.* And note that M3 and M4 were both caught by a worker who stopped and said the premise was wrong —
which is why §3.6's self-audit demand and §7's escalation list are not optional decoration.

---

## 9. Start here

1. Confirm you have read this and the five attached files.
2. Run the **handover audit** (§1). About an hour. Report what you verified and what, if anything, failed.
3. **Raise D18 immediately** — one canonical copy of the planning files, before Day 1.
4. Chase the open items in §2: which five days, and D11.
5. Send **Dispatch #2**, which is already written and waiting in `DISPATCHES_2026-07-30.md`.
6. Then write the Day 1 dispatches: **T1 paper selection** and **T2 runnability probe** — and remember T2 is
   the one that decides whether this project is a reconstruction or a book report. It must run the repo's own
   example end to end and **time it**, not read the code and conclude it will work.
