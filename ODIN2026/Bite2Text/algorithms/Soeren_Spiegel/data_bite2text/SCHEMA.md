# Bite2Text canonical chart schema (v1.2)

One JSON record per case, extracted from ALL English reports of that case in a single
pass (IOS-based and photo-based together; per TF4 lesson, never per-report).

v1.1 incorporated the first pilot-gate review: hedge qualifier, posterior crossbite
scope, arch-scoped curves, duplicate-report dedup, contested tooth facts, and the
binding merge rules. v1.2 (second gate round) makes dedup MECHANICAL — the report
dump now prints deterministic VOTE GROUPS per case — and adds rules 17–22.

## Global conventions

- **Stated facts only.** Record a field only if at least one report states it. Omit
  unstated fields entirely — "unmentioned ⇒ normal" defaults are applied later in
  code, never at extraction time.
- Every recorded field is an object `{"v": <value>, "src": [<src>, ...]}` where src is
  `"ios:<rid>"` or `"photo:<rid>"` (e.g. `"ios:12_512"`).
  - `src` lists ONLY reports stating the exact recorded value. A report filed in
    merge_notes under a different value must NOT appear in that field's src.
  - Optional `"hedged": true` on any field object when the statement is hedged
    ("may be", "possibly", "appears", "seems", "not easy to say"); keep the hedge
    wording verbatim in `note`. A hedged tooth must never appear in a teeth array
    (recession.teeth, crossbite.teeth, ...) as if definite — put it in `note`.
  - Optional `"note": "<verbatim qualifier or wording the enum cannot hold>"`.
- `"not_assessable"` is a first-class value in EVERY enum below (including spacing and
  all soft_tissue fields). Never coerce it to normal.
- FDI: non-dotted 2-digit strings. Permanent 11–18, 21–28, 31–38, 41–48; deciduous
  51–55, 61–65, 71–75, 81–85 (pediatric cases are common — deciduous numbers are
  valid and must not be "corrected").
- Values are closed enums; if a report says something the enum cannot express, pick
  the closest value and put the report's wording in `note`. Anything no field can
  hold goes verbatim into `other_findings` — never drop a clause.

## Record

```
{
  "case_id": "F1979",
  "sources": {"ios": ["12_512", ...], "photo": ["12_2988", ...]},   // all EN rids used
  "dentition": {"v": "permanent|mixed|deciduous"},
      // phase wording ("early mixed", "mixed dentition 2") verbatim in note;
      // annotator disagreement on phase goes to merge_notes.
  "transverse": {
    "maxilla":      {"v": "normal|constricted|expanded|not_assessable"},
    "crossbite":    {"v": "none|present", "teeth": ["17", ...],
                     "region": "right|left|bilateral|anterior|posterior|unstated"},
      // region is allowed (and required when stated) on v="none" to scope the
      // negation: "no posterior crossbite" => v="none", region="posterior".
    "scissor_bite": {"v": "none|present", "teeth": [...], "region": "as crossbite (optional)"}
  },
  "vertical": {
    "overbite": {"v": "normal|increased|reduced|open_bite|edge_to_edge|not_assessable"}
    // "deep bite" => increased; anterior/lateral open bite detail goes in note
  },
  "sagittal": {
    "molar_class_right":  {"v": "I|II|III|not_assessable", "degree": "full|end_to_end|half|quarter|three_quarter|tendency|unstated"},
    "molar_class_left":   { same },
    "canine_class_right": { same },
    "canine_class_left":  { same },
    "overjet": {"v": "normal|increased|reduced|zero|reverse|not_assessable"}
  },
  "midline": {
    "upper": {"v": "coincident|deviated_right|deviated_left|deviated|not_assessable"},
    "lower": { same }
    // "midlines are coincident" => both coincident; "deviated relative to each other"
    // with no side/arch => set BOTH to "deviated" with a note
  },
  "curve_spee":   {"v": "normal|increased|flat|not_assessable", "arch": "upper|lower|both|unstated"},
  "curve_wilson": {"v": "normal|increased|flat|not_assessable", "arch": "upper|lower|both|unstated"},
      // "lower curve of Spee increased" => arch="lower". Never merge an upper-arch
      // statement with a lower-arch one as if they were the same fact.
  "arch_upper": {
    "crowding": {"v": "none|mild|moderate|severe|not_assessable"},
    "spacing":  {"v": "none|present|diastema|not_assessable"}
  },
  "arch_lower": { same },
  "teeth": {                    // EXCEPTIONS ONLY — teeth explicitly mentioned
    "<FDI>": {"findings": ["absent|unerupted|erupting|impacted|retained_deciduous|rotated|displaced|infra_occlusion|crossbite|scissor_bite|sealed|caries_suspect|restored|prosthetic_crown|worn|fractured|other"],
               "note": "<free text>", "src": [...], "contested": true?}
      // "restored" = fillings/restorations; crowns use "prosthetic_crown";
      // "worn" = attrition/worn incisal edges. Reserve "other" for findings no
      // enum value covers. "contested": true marks a tooth whose facts conflict
      // (see merge rules 12-13).
  },
  "soft_tissue": {              // mostly photo-derived
    "plaque":       {"v": "none|present|not_assessable"},
    "staining":     {"v": "none|present|not_assessable"},
    "gingival_inflammation": {"v": "none|present|not_assessable"},
    "recession":    {"v": "none|present|not_assessable", "teeth": [...]},
    "hygiene":      {"v": "good|fair|poor|not_assessable"},
    "other":        {"v": "<free text>"}
  },
  "appliances": [ {"v": "<free text, verbatim attributes only>", "src": [...]} ],
  "other_findings": [ {"v": "<free text — anything the schema cannot hold>", "src": [...]} ],
  "merge_notes": [ {"field": "...", "values": {"<enum value>": [srcs], ...}, "note": "..."} ]
      // field may be a schema path, "<field>.degree" for degree-only disagreements,
      // or "teeth.<FDI>". Value keys MUST be members of that field's enum — never
      // verbatim report wording, never invented paths.
}
```

## Extraction & merge rules (binding — from the pilot gate review)

Value selection
1. Merge across reports by majority over VOTE GROUPS stating the field; ties break
   toward IOS for occlusal/geometric fields, toward photo for tooth-surface/soft-tissue.
2. The recorded v must be a value at least one report literally supports AND that wins
   the tally. Never emit a grade outside the stated range ("moderate" 2/3 vs
   "moderate-to-severe" 1/3 ⇒ "moderate", never "severe").
3. Range wordings ("mild/moderate", "moderate-to-severe") map to the LOWER named grade
   with the verbatim wording in note.
4. **Vote groups are printed in the report dump header — use them verbatim, never
   re-derive them.** Reports in one group are ONE vote (some cases duplicate one
   annotator's text across dirs). Any "majority (N/M)" wording in notes must count
   GROUPS: M = number of groups stating the field. When a collapse changes a tally,
   say so in note.
5. Ties the modality rule cannot break (tied groups of the same modality), and 3-way
   splits: prefer the non-"normal" value and write "unresolved tie" in note, so the
   merged record stays distinguishable from the unmentioned-implies-normal default.
   When the members of ONE group disagree with each other on a field (one text says
   normal, its twin says cannot be assessed), that group casts no clean vote on that
   field — treat the field as an unresolved tie and apply the non-normal preference.

Coverage
6. Before emitting, walk each source report clause by clause and confirm every clause
   is represented in some field. Anything unmapped goes verbatim into other_findings.
   An empty other_findings for a report containing a parenthetical clinical remark
   ("the torque of the posterior teeth is positive") is a bug.
7. A diastema or spacing mentioned anywhere — including as the reason for another
   finding ("midlines not definable due to the wide interincisal diastema") — must set
   arch_<arch>.spacing, not only a note.
8. Named tooth groups with a fixed FDI expansion ("the third molars" ⇒ 18/28/38/48,
   "the maxillary first molars" ⇒ 16/26) expand into teeth entries; only genuinely
   arch- or sextant-level statements stay free text.
9. Set crossbite.region from named teeth when the FDI code fixes the side (17 ⇒
   "right"); reserve "unstated" for statements with no side-determining content.

Contradictions
10. Refinements of the same value ("anterior diastemas" vs "presence of spaces",
    "markedly increased" vs "increased") are NOT contradictions — log merge_notes only
    for mutually exclusive values.
11. Degree-only disagreements (both say Class II, one "full" one "end-to-end") log
    under "<field>.degree", not the field itself.
12. A report's group-level eruption/presence INVENTORY overrides its own conflicting
    tooth-level clause. Pick the tooth that satisfies BOTH the inventory and the other
    reports (inventory says the maxillary first molars are erupted + another report
    names 46 as unerupted ⇒ the unerupted tooth is 46, NOT 16), set "contested": true
    on the affected tooth, and log merge_notes. Never record a tooth as
    absent/unerupted solely because a report cites it as the reason a class is
    unassessable — first check it against that report's own inventory.
13. When one report contradicts itself (a tooth both missing and restored), record
    both findings, set "contested": true with the conflict in that tooth's note, and
    log it in merge_notes with both srcs being the same rid.

v1.3 amendments (post-verification rulings — applied corpus-wide by sweep,
data_bite2text/sweep_fixes.log)
5a. A pure-photo group's "cannot be assessed (from these photos)" is a MODALITY
    LIMITATION, not a clinical vote: it never outvotes an IOS-backed stated value
    (rule 1 governs; rule 5's non-normal preference does not apply to such ties).
5b. Hedge-then-pivot wordings ("difficult to define… but appears centered"): the
    hedged CONCLUSION is the group's vote, marked "hedged": true — rule 19 never
    converts a stated finding into not_assessable.
5c. A clean vote for "normal" from one group beats a self-inconsistent group's
    ambiguity: rule 5's non-normal preference applies only when NO clean vote
    exists on the field.

Inference limits
14. Record no attribute the text does not state ("splint canine-to-canine" must not
    become "fixed splint").
15. Infer dentition="mixed" only when a deciduous tooth, a named mixed/early-mixed
    dentition, or an eruption stage appears; one unerupted permanent tooth in an
    otherwise permanent dentition is not sufficient. Phase wordings ("early mixed",
    "mixed dentition 2") go verbatim in note, never as merge_note keys; if all groups
    agree on the enum value there is NO merge_note (rule 10).
16. Sides are the PATIENT's right/left; "bilaterally" fills both sides' slots.

Src & qualifier discipline (v1.2)
17. src lists only reports whose literal wording maps to the recorded value. For
    refinement pairs ("spaces" vs "anterior diastemas", "increased" vs "markedly
    increased"): record the more specific value, keep the weaker-wording report in
    note, drop it from that field's src. Per-tooth src lists only the reports stating
    THAT tooth's findings; when different findings come from different reports, give
    the per-finding attribution in the tooth's note.
18. Scoped qualifiers (region, arch) must be backed by the srcs: never narrow a
    negation below the broadest stated scope ("no crossbite" + "no posterior
    crossbite" ⇒ region "unstated", narrower wording in note). "Lateral crossbite"
    means posterior. Never set an arch on a curve when a listed src states no arch —
    scope to what the srcs share, put narrower wordings in note.
19. Hedge scan per FIELD, over the whole record (appears / seems / may be / might /
    possibly / would appear / not easy to say): set "hedged": true when the majority
    of the field's supporting groups hedge; when only some hedge, name them in note.
    Apply to transverse, overbite, overjet too — not only curves and soft tissue.
20. Class softeners ("weak Class I", "mild Class II", "tendency toward") map to
    degree="tendency" with the wording verbatim in note — never degree="unstated"
    when a softener exists.
21. Never emit empty arrays ("teeth": []) — omit the key. Every recorded field object
    carries src, including soft_tissue.other and appliances entries.
22. A tooth-level fact may never live only inside another field's note: expand named
    groups into teeth entries even when the clause is the reason for another finding
    ("midlines not definable due to the wide interincisal diastema" still sets
    spacing; "the mandibular incisors have not yet fully erupted" still creates
    erupting entries for 31/41).
