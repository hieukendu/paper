# OpenReview / ARR Submission Draft (Internal Only)

This preparation file is not part of the anonymous submission. It reflects the ARR guidance and area-keyword list inspected on 2026-07-30. Authors must check the live form and current call before submitting.

## Submission identity

- Proposed title: **ViPragSent-Final: A Development-Selected Ensemble for Vietnamese Pragmatic Sentiment Evaluation**
- Submission type: Long paper
- Target venue: EACL 2027 Main Conference
- ARR cycle: `[ARR CYCLE]`
- Paper language: English
- Review status: anonymous/double-blind
- Canonical PDF: `artifact/vipragsent-repro/manuscript/latex/main.pdf`
- PDF SHA-256: `04b5be0d80b605a216d0935fda423008935f753f5231895fe0e08c9a303c0ffb`
- Current PDF page count: 14
- Page reduction: deferred to the authors' separate supervisor-led phase

## Abstract

Pragmatic interpretation in Vietnamese-language text can depend on signals beyond surface polarity. We evaluate six pragmatic phenomena in the fixed ViPragSent setting, which has a documented social-media source component and a private-data boundary. ViPragSent-Final is a frozen label-wise ensemble selected with five-fold development out-of-fold evidence: it combines retained ViSoBERT and PhoBERT sources, probability blends for irony and code-switching, and a targeted mocking component. On one post-freeze canonical-test evaluation, it reaches 84.39 macro pragmatic F1, 1.56 points above Vistral-7B, the strongest complete best-tuned baseline. The aggregate lead does not establish strict per-label dominance: the final system remains 0.40 points below the strongest irony result, 0.11 below the strongest idiom/figurative-language result, and 0.13 below the strongest code-switching result. Stored predictions, configuration, and artifact hashes are traceable, but this is not an independent training reproduction; raw source text remains private.

## Keywords and ARR areas

- Keywords: Vietnamese NLP; pragmatic sentiment; social media; multi-label evaluation; development-selected ensemble; reproducibility.
- Primary ARR area: **Resources and Evaluation** Ã¢â‚¬â€ the core contribution is a provenance-bounded benchmark, fixed evaluation, and result traceability.
- Secondary ARR area: **Discourse and Pragmatics** Ã¢â‚¬â€ the task covers pragmatic inference phenomena beyond surface polarity.
- Secondary ARR area: **Sentiment Analysis, Stylistic Analysis, and Argument Mining** Ã¢â‚¬â€ the empirical target is pragmatic sentiment and affective interpretation.
- Optional secondary area: **Multilingualism and Cross-Lingual NLP** Ã¢â‚¬â€ Vietnamese, code-switching, and a less-resourced-language resource are central.

The names above are the current ARR area names; authors should select only those exposed by the live form.

## Form-ready contribution summary

ViPragSent defines a fixed six-label Vietnamese pragmatic-sentiment evaluation with separate polarity and emotion auxiliaries. It documents a 12,000-record post-adjudication corpus, an 8,000/2,000/2,000 split, provenance and private-data boundaries, and a comparative evaluation across encoder, social-media, multilingual, prompted, and instruction-tuned families. ViPragSent-Final is a development-only five-fold-OOF-selected, frozen label-wise ensemble, not a claim of architectural novelty. On its one post-freeze canonical-test evaluation it achieves 84.3883 macro pragmatic F1, +1.5633 over the strongest complete best-tuned baseline, while remaining below the label leaders on irony, idiom/figurative language, and code-switching. Stored predictions, configurations, thresholds, and hashes provide artifact-level traceability; raw source text remains private and no independent full training rerun is claimed.

## Anonymous artifact statement (reviewer-facing)

Model checkpoints, configurations, predictions, and evaluation artifacts are available through anonymized supplementary materials. Direct repository identifiers are withheld during double-blind review. Raw and processed source text are private and are not redistributed. `[ANONYMIZED_ARTIFACT_LINK]` `[ANONYMIZED_SUPPLEMENTARY_ARCHIVE]`

## INTERNAL ONLY Ã¢â‚¬â€ remove from anonymous OpenReview fields

- GitHub repository: `https://github.com/hieukendu/paper`
- Experiment checkpoints: `https://huggingface.co/Thundergod2007/vipragsent-experiment-checkpoints`
- Vistral QLoRA: `https://huggingface.co/Thundergod2007/vipragsent-vistral-7b-qlora`
- Sailor QLoRA: `https://huggingface.co/Thundergod2007/vipragsent-sailor-7b-qlora`

Camera-ready wording: Ã¢â‚¬Å“Code, public model artifacts, configurations, predictions, evaluation artifacts, and documentation are available at the project repository and model archives; raw source text remains unavailable because of the documented data-access boundary.Ã¢â‚¬Â Authors must validate all links, licences, and access wording before use.

## Responsible NLP Research Checklist draft

The wording below reproduces the current checklist questions. Proposed answers must be approved by the authors in the live form.

| Question (current wording) | Proposed response | Explanation and manuscript evidence | Author confirmation |
| --- | --- | --- | --- |
| A1. Did you describe the limitations of your work? | YES | Section 8 states fixed-benchmark, private-data, selection, fairness, and reproducibility limits. | YES |
| A2. Did you discuss any potential risks of your work? | YES | Section 9 discusses privacy, profiling, surveillance, moderation, representation, and human oversight. | YES |
| B1. Did you cite the creators of artifacts you used? | YES | Sections 2--4 and the audited bibliography cite model, dataset, and external-diagnostic sources. | YES |
| B2. Did you discuss the license or terms for use and / or distribution of any artifacts? | NO, with justification | Sections 3, 8, and 9 disclose unresolved source authorization and licence questions; no permission is claimed. | **YES** |
| B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? | NO, with justification | The paper limits claims and no raw redistribution; authors must confirm terms for source and diagnostic assets. | **YES** |
| B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? | YES, bounded | Sections 3 and 9 report private-data handling and aggregate-only reporting; do not extend this answer beyond repository evidence. | **YES** |
| B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? | YES, bounded | Sections 3--4 and appendices describe Vietnamese language, source strata, labels, and scope; coverage limits remain explicit. | YES |
| B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? | YES | Section 3 reports 12,000 records, source composition, prevalence, and 8,000/2,000/2,000 split. | YES |
| C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? | NO, with justification | Section 4 reports available hardware/software and model-scale limits; exact historical total compute is not fully recorded. | **YES** |
| C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? | YES | Section 4 and final-selection artifacts document training settings, OOF selection, thresholds, and freeze. | YES |
| C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? | YES, bounded | Sections 4--5 identify seed scope, bootstrap scope, and missing aligned-comparator predictions. | YES |
| C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation, such as NLTK, Spacy, ROUGE, etc.), did you report the implementation, model, and parameter settings used? | YES, bounded | Section 4 and artifact manifests record the available implementation/configuration details. | YES |
| D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? | YES, bounded | Annotation guidelines and Section 3/Appendix A describe the protocol; authors must ensure attachment/publication is permitted. | **YES** |
| D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participantsÃ¢â‚¬â„¢ demographic (e.g., country of residence)? | NO, with justification | The repository does not establish recruitment or compensation details; do not invent them. | **YES** |
| D3. Did you discuss whether and how consent was obtained from people whose data youÃ¢â‚¬â„¢re using/curating? | NO, with justification | The paper explicitly does not claim consent or resolved source authorization. | **YES** |
| D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? | NO, with justification | No approval/exemption evidence is recorded. | **YES** |
| D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? | NO, with justification | Such population characteristics are not established by the repository. | **YES** |
| E1. If you used any AI assistants, did you include information about your use? | YES, pending approval | Use the disclosure below; authors remain responsible for the final content. | **YES** |

## AI writing-assistance disclosure

### Review-form version

AI-assisted tools supported language editing, manuscript organization, code and consistency review, and citation checking. The authors reviewed and approved the final manuscript, remain responsible for all claims and citations, and did not list any AI system as an author. No experimental result was invented or altered through AI assistance.

### Internal detailed record

AI assistance was used for language editing, manuscript organization, code and configuration review, citation/source checking, numerical consistency checks, and PDF-format inspection. Repository artifacts, code, and official primary sources were used to ground changes. Authors must independently approve the final wording, verify citations, and decide the venue-specific disclosure text. This record does not assert that AI performed experiments, obtained permissions, or became an author.

## Ethics and limitations form answers

The paper reports a fixed Vietnamese pragmatic-sentiment benchmark with private/social-media-derived source material and author-created context-augmented derivatives. Raw text and identifiable examples are not released. Source authorization, licence, and third-party diagnostic terms remain unresolved and must not be represented as complete. The final system is development-selected and evaluated once after a frozen configuration; results therefore do not prove universal architecture superiority or strict per-label dominance. Potential misuse includes automated moderation, profiling, surveillance, and misrepresentation. Any operational use requires human oversight and governance review.

## Author and reviewer-registration placeholders

- `[AUTHOR 1 FULL NAME]`
- `[AUTHOR ORDER CONFIRMATION]`
- `[OPENREVIEW PROFILE STATUS]`
- `[REVIEWER REGISTRATION STATUS]`
- `[CORRESPONDING AUTHOR]`
- `[CONFLICT DOMAINS]`

Authors must confirm all names, order, affiliations, email addresses, conflicts, reviewer-registration obligations, and ARR/EACL cycle data in the live system.

## Declarations placeholders

- Conflicts of interest: `[AUTHOR CONFIRMATION REQUIRED]`
- Funding and acknowledgements: `[AUTHOR CONFIRMATION REQUIRED]`
- Prior/parallel submission and preprint status: `[AUTHOR CONFIRMATION REQUIRED]`
- Data-access statement: `[AUTHOR/LEGAL CONFIRMATION REQUIRED]`
- Code/model availability: `[ANONYMIZED REVIEW LINK OR AUTHOR-APPROVED NO-LINK WORDING]`

## Final author confirmation checklist

- Confirm the live ARR/EACL call, area names, page rules, and deadline/cycle.
- Approve the Responsible NLP answers and AI-use disclosure.
- Confirm source-text authorization, VIVID-derived-material permission, and third-party diagnostic terms.
- Confirm author list/order, conflicts, funding, acknowledgements, and OpenReview/reviewer registration.
- Choose and test a genuinely anonymous artifact-hosting mechanism.
- Perform the separately deferred page-reduction phase before any page-limit-dependent submission.
