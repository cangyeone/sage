# Nature Communications Manuscript Draft

Working title:

**Autonomous evidence-driven seismology reveals testable earthquake-system hypotheses from data, literature and executable analysis**

中文定位：

这是一篇偏 Nature Communications 风格的论文草案，核心不是把 SAGE 写成一个普通软件系统，而是把它包装成一种“可追溯、可执行、可自我调试的地震学科研智能体”范式。真正能否达到 NC 级别，取决于后续是否能用 SAGE 在一个真实数据集上挖掘出新的、可验证的地震学问题或机制，例如：

- 某一区域地震序列是否存在此前未被注意的触发机制转换；
- 自动目录中的不确定性是否揭示出传统确定性目录忽略的构造或流体控制；
- 深度、震相残差、b 值、迁移速度、应力扰动之间是否存在新的组合型证据链；
- 注水、断层几何、速度结构和微震迁移之间是否能形成可检验的新假说。

本文稿目前采用“平台 + 发现范式 + 案例占位”的写法。凡涉及真实结果的位置均用 `[RESULT TO FILL]` 标记，后续应由 evidence-geo-agent 的 evidence table、代码运行产物、图件、统计结果和文献证据自动填入。

---

## Title

**Autonomous evidence-driven seismology reveals testable earthquake-system hypotheses from data, literature and executable analysis**

## Short Title

Autonomous evidence-driven seismology

## Authors

Yuziye [Surname]\*, [Co-authors to be added]

## Affiliations

[Institutional affiliations to be added]

## Correspondence

Correspondence to: [email]

---

## Abstract

Modern earthquake monitoring is being transformed by automated detection, deep-learning phase picking and continuously growing waveform archives. Yet the scientific interpretation of these data remains fragmented: catalog generation, uncertainty assessment, literature synthesis, statistical testing and figure production are often performed as separate, weakly connected steps. This separation makes it difficult to trace how a geological interpretation emerges, to identify which assumptions control the conclusion, and to convert unexpected patterns into testable earthquake-system hypotheses.

Here we present **SAGE**, an autonomous evidence-driven seismology agent that couples retrieval-augmented literature reasoning, executable coding, skill-based domain workflows, multimodal document analysis and uncertainty-aware evidence tracking. Given a research question and heterogeneous inputs, SAGE iteratively retrieves relevant knowledge, writes and debugs analysis code, generates intermediate figures and statistics, scores competing hypotheses, and produces a traceable manuscript-style interpretation. Unlike conventional chat-based assistants, SAGE stores every intermediate claim, figure and test result as an evidence record with provenance, confidence and polarity.

We demonstrate the framework on `[STUDY REGION / DATASET TO FILL]`, where SAGE identifies `[NEW SCIENTIFIC QUESTION TO FILL]` from automated earthquake catalogs, waveform-derived measurements and prior literature. The agent formulates competing hypotheses involving `[HYPOTHESIS A]`, `[HYPOTHESIS B]` and `[HYPOTHESIS C]`, then tests them using `[STATISTICAL TESTS / CATALOG ANALYSES / WAVEFORM ANALYSES TO FILL]`. The resulting evidence chain suggests that `[KEY FINDING TO FILL]`, revealing `[BROADER IMPLICATION TO FILL]`.

Our results show that autonomous scientific agents can move beyond answer generation toward reproducible hypothesis discovery in earthquake science, provided that reasoning is grounded in executable analysis and explicitly tracked evidence. This approach offers a path toward trustworthy automated catalogs, auditable geophysical interpretation and accelerated discovery in data-rich seismology.

---

## Main Text

### Introduction

Earthquake science is entering an era in which data acquisition is no longer the primary bottleneck. Dense seismic arrays, cloud-hosted waveform archives, deep-learning phase pickers and automated association algorithms can produce earthquake catalogs at unprecedented scale. These advances have exposed smaller events, richer sequence structures and subtle spatiotemporal patterns that were previously inaccessible. However, the scientific conversion of such data into geological understanding remains difficult.

Three barriers are particularly persistent. First, automated catalogs are often treated as deterministic products, even though detection thresholds, phase-picking uncertainty, association ambiguity and location uncertainty can strongly influence downstream interpretations. Second, the reasoning process that links raw data, derived measurements, statistical tests and geological hypotheses is rarely represented explicitly. Third, domain expertise is distributed across papers, software packages, notebooks, figures and local conventions, making it hard to reproduce how a conclusion was reached or to systematically explore alternative explanations.

Large language models and AI agents offer a possible route through this bottleneck, but naive use of conversational models is insufficient for scientific discovery. A model that only produces fluent text can hallucinate unsupported claims, overlook numerical inconsistencies, or fail to execute the analysis required to validate an idea. For earthquake science, a useful agent must do more than answer questions: it must inspect data, write and debug code, search literature, evaluate competing hypotheses, generate figures, preserve intermediate outputs and expose the evidence behind each conclusion.

Here we introduce SAGE, a seismology AI-guided engine designed around evidence-first scientific reasoning. SAGE integrates five capabilities into a single workflow: retrieval from local and online knowledge sources, skill-guided domain programming, executable analysis with self-debugging, multimodal interpretation of papers and figures, and structured hypothesis scoring. The system is designed not only to help users run existing workflows, but also to uncover new scientific questions from the tension between data, literature and intermediate results.

We use SAGE to address a central question for modern earthquake monitoring: **Can an autonomous, evidence-tracking agent discover and test new earthquake-system hypotheses from automated catalogs and waveform-derived data?** We show that the answer depends critically on preserving uncertainty and intermediate evidence. In a case study from `[REGION TO FILL]`, SAGE identifies `[NEW QUESTION TO FILL]`, generates `[NUMBER]` intermediate analyses, evaluates `[NUMBER]` competing hypotheses, and produces a reproducible evidence chain linking observations to interpretation.

The contribution of this study is therefore twofold. Scientifically, we provide a framework for extracting testable earthquake-system hypotheses from complex monitoring data. Methodologically, we demonstrate a pattern for trustworthy AI-assisted geoscience: every claim must be grounded in retrievable evidence, every quantitative statement must be tied to executable code, and every final interpretation must preserve the intermediate materials from which it emerged.

---

## Results

### SAGE couples reasoning, evidence tracking and executable seismology

SAGE is organized as an evidence-driven scientific agent rather than a single-purpose chatbot. A user begins with a research question, dataset, paper, waveform archive, catalog or geological hypothesis. The agent then enters an iterative loop consisting of five stages: question decomposition, evidence retrieval, executable analysis, hypothesis update and synthesis.

In the first stage, SAGE converts the user request into a set of candidate scientific questions and operational tasks. For example, a broad question such as whether a microearthquake sequence is fluid-driven may be decomposed into tests of event migration, depth distribution, b-value changes, waveform similarity, temporal correlation with injection, and consistency with published stress or permeability models.

In the second stage, the agent retrieves evidence from multiple sources. These include local knowledge bases, uploaded papers, web-search results, seismic catalogs, waveform metadata, prior code outputs and skill documents. Each retrieved item is converted into an evidence record containing its source, type, confidence, polarity and relationship to current hypotheses.

In the third stage, SAGE writes and executes code. This step distinguishes the system from text-only reasoning. The coding engine can read waveform files, process catalogs, compute statistics, generate figures, compile LaTeX, and run mini-tests to debug its own functions. Intermediate scripts, standard output, figures and tables are preserved as analysis artifacts.

In the fourth stage, SAGE updates competing hypotheses. Rather than producing a single conclusion immediately, the agent maintains multiple possible explanations and scores them against accumulated evidence. Evidence can support, contradict or remain neutral with respect to a hypothesis.

In the final stage, the system generates a report or manuscript draft in which all claims are linked to evidence records and intermediate artifacts. This creates a transparent chain from data to interpretation.

**Result to fill:** Quantify the agent workflow on the chosen case study:

- Number of evidence records generated: `[N]`
- Number of code runs: `[N]`
- Number of self-debugging iterations: `[N]`
- Number of figures/tables generated: `[N]`
- Number of hypotheses evaluated: `[N]`
- Converged hypothesis score separation: `[DELTA]`

### Autonomous exploration exposes candidate earthquake-system questions

The central scientific output of SAGE is not merely a processed dataset, but a ranked set of testable questions. In the `[REGION / SEQUENCE TO FILL]` case study, the agent began from `[INITIAL USER QUESTION TO FILL]` and generated candidate scientific questions including:

1. Does the earthquake sequence exhibit statistically significant migration consistent with fluid-pressure diffusion?
2. Do catalog location uncertainties alter the apparent geometry of the sequence?
3. Are changes in b value or magnitude-frequency behavior temporally correlated with operational or tectonic forcing?
4. Do waveform similarity clusters indicate repeated rupture on persistent asperities?
5. Are apparent depth trends robust to phase-picking and velocity-model uncertainty?
6. Is the observed sequence better explained by direct triggering, delayed triggering, aseismic slip, or catalog incompleteness?

The agent then filtered these questions using three criteria: whether relevant data were available, whether the question was not already answered in the retrieved literature, and whether the hypothesis could be tested with executable analysis. This produced the focal scientific problem:

> **[NEW SCIENTIFIC QUESTION TO FILL]**

This question is suitable for a high-impact study if it satisfies three conditions: it reveals a previously underexplored mechanism, it can be tested quantitatively, and it changes how automated catalogs should be interpreted in the study region.

**Result to fill:** Insert the final discovered question and explain why it is novel relative to the retrieved literature.

### Uncertainty-aware catalogs change the geological interpretation

A key design principle of SAGE is that automated catalogs should not be treated as fixed truth. Each event inherits uncertainty from detection, picking, association, velocity model and location. Conventional workflows often propagate only the final hypocenter and magnitude, causing downstream analyses to ignore ambiguity that may be geologically meaningful.

In the case study, SAGE tests how catalog uncertainty affects `[INTERPRETATION TARGET TO FILL]`. The agent compares interpretations based on:

- deterministic hypocenters;
- location probability clouds or bootstrap relocations;
- phase-pick uncertainty;
- alternative velocity models;
- event subsets filtered by quality metrics;
- sensitivity to magnitude completeness.

This analysis asks whether the inferred geological pattern is robust or an artifact of catalog construction. For example, a narrow planar distribution may disappear when location uncertainty is considered, whereas a migration front may remain stable across bootstrap realizations. Conversely, uncertainty may reveal that two apparently separate clusters are not statistically separable.

**Result to fill:** Report the uncertainty-sensitive finding:

`[Example placeholder: Accounting for location uncertainty reduces the apparent dip of the event plane from X degrees to Y degrees, but preserves a NE-directed migration signal at Z km/day.]`

### Executable evidence discriminates among competing hypotheses

SAGE evaluates multiple hypotheses rather than directly committing to one interpretation. For the selected case, the initial hypothesis set is:

- **H1: Fluid-pressure diffusion.** Seismicity migration and temporal clustering are controlled by pore-pressure perturbations.
- **H2: Aseismic-slip loading.** Earthquakes are triggered by stress transfer from aseismic deformation.
- **H3: Tectonic stress release.** The sequence reflects background tectonic loading without a dominant transient forcing.
- **H4: Catalog artifact.** The apparent pattern is produced by detection threshold, station geometry, phase-picking bias or location uncertainty.
- **H5: Mixed mechanism.** Multiple processes operate at different stages of the sequence.

For each hypothesis, SAGE generates targeted tests. Fluid diffusion is tested using migration distance versus time, diffusion-like envelopes, temporal correlation with injection or hydrological records, and spatial relation to permeable structures. Aseismic slip is tested using migration asymmetry, moment release, waveform similarity, depth evolution and possible geodetic constraints. Catalog artifact hypotheses are tested through quality metrics, station coverage, synthetic perturbation and robustness analysis.

The agent records each test as evidence. A simplified evidence table may contain:

| Evidence ID | Evidence | Source | Supports | Contradicts | Confidence |
|---|---|---|---|---|---|
| E001 | `[Observation about migration]` | `[Figure/Table]` | H1 | H4 | `[score]` |
| E002 | `[Observation about uncertainty]` | `[Bootstrap test]` | H4 or H5 | H1 | `[score]` |
| E003 | `[Literature constraint]` | `[Paper]` | H2 | H3 | `[score]` |
| E004 | `[Waveform similarity result]` | `[Code output]` | H2/H5 | H1 | `[score]` |

**Result to fill:** Insert the final hypothesis scores and the evidence records that drive the ranking.

### Intermediate artifacts become scientific material rather than discarded by-products

Traditional analysis workflows often discard intermediate products such as failed scripts, diagnostic plots, sensitivity tests or alternative figures. SAGE treats these artifacts as part of the scientific record. This is important because scientific insight often emerges from intermediate anomalies: an unexpected residual pattern, a failed model assumption, a cluster visible only after filtering, or a discrepancy between literature and data.

In the case study, the agent produces intermediate materials including:

- catalog maps colored by time, depth, magnitude and uncertainty;
- cross-sections along candidate fault orientations;
- magnitude-frequency and completeness analyses;
- STA/LTA or waveform-trigger diagnostics;
- phase-picking residual distributions;
- event migration curves;
- waveform similarity matrices;
- hypothesis score trajectories;
- tables of conflicting evidence.

These artifacts are not supplementary debris; they are the basis for deciding which scientific question is worth pursuing. SAGE therefore writes them directly into the manuscript as figures, extended data or supplementary tables.

**Result to fill:** List the intermediate artifacts that led to the final scientific question.

### A manuscript can be generated from the evidence graph

Once hypotheses and evidence records converge, SAGE generates a manuscript draft. The manuscript is not a free-form language model output. Instead, the structure is assembled from:

1. the final question;
2. the ranked hypotheses;
3. evidence records;
4. figures and tables;
5. code-generated statistics;
6. retrieved literature;
7. unresolved uncertainties.

This produces a manuscript in which claims can be traced backward. For example, a sentence such as “the sequence migrates northeast at approximately `[VALUE]` km/day” is linked to the script that computed the migration rate, the figure that visualizes it, and the catalog subset used for the calculation.

The output can be rendered as Markdown, HTML, LaTeX or PDF. This closes the loop from discovery to communication and allows human researchers to edit, challenge or rerun the analysis.

---

## Discussion

The increasing scale of automated earthquake monitoring creates a paradox. More events are detected, but the path from detection to scientific understanding becomes harder to audit. SAGE addresses this problem by making the interpretive process explicit. The agent does not replace seismologists; rather, it expands the surface area of scientific exploration by connecting data, code, literature and uncertainty into a coherent evidence graph.

The most important implication is that autonomous agents can help discover questions, not merely answer them. A human researcher may ask whether a sequence is induced, but the agent may reveal that the more precise question concerns whether location uncertainty masks a transition between diffusion-like migration and fault-controlled rupture. This reframing is often where scientific novelty begins.

A second implication is that uncertainty is not a post-processing detail. In automated catalogs, uncertainty can determine whether a pattern is real, whether a fault plane is resolved, whether migration is significant, and whether competing mechanisms can be separated. SAGE therefore treats uncertainty as first-class evidence.

A third implication concerns reproducibility. Scientific reasoning is usually preserved only in final prose, while the intermediate choices that shaped the conclusion remain scattered across notebooks and memory. By storing intermediate artifacts and evidence records, SAGE makes the discovery pathway inspectable.

Several limitations remain. First, the quality of the output depends on the quality of available data, local skills and knowledge bases. Second, literature retrieval can be incomplete or biased by search access. Third, code generation requires robust sandboxing, mini-tests and debugging to avoid silent numerical errors. Fourth, hypothesis scoring remains partly heuristic and should be calibrated against expert judgment. Finally, manuscript generation should be treated as an editable scientific draft, not an automatic publication.

Despite these limitations, evidence-driven scientific agents offer a practical route toward trustworthy AI in earthquake science. The central requirement is not that an AI model be universally correct, but that every step of its reasoning be tied to retrievable evidence and executable analysis.

---

## Methods

### System overview

SAGE consists of a chat interface, intent router, retrieval system, skill manager, coding engine, evidence-driven geoscience agent and manuscript generator. The system accepts natural-language questions, uploaded documents, waveform files, catalogs, local workspace paths and optional online search results.

The intent router classifies each request into question answering, code execution, chained literature-to-code analysis, or general conversation. During streaming responses, the system can hand off from question answering to code execution when the reasoning process identifies that executable analysis is required.

### Evidence record schema

Each evidence record contains:

- unique evidence ID;
- textual summary;
- source path or URL;
- evidence type: text, figure, table, code output, waveform analysis, catalog statistic or literature;
- confidence score;
- polarity with respect to each hypothesis;
- producing tool call;
- associated figure or table;
- timestamp.

This schema allows final conclusions to be traced to their supporting evidence.

### Retrieval and skill grounding

SAGE retrieves context from local knowledge bases, uploaded papers, skill documents and optional web search. Skill documents define domain-specific workflows such as waveform reading, STA/LTA triggering, catalog mapping, 3D terrain plotting, uncertainty visualization and LaTeX compilation.

Retrieved skills are injected into the coding engine so that generated code follows local conventions and uses tested helper functions. A `.md` file under `seismo_skill/skills/` is treated as a standalone skill, whereas a folder under `seismo_skill/skills/` is treated as a composite skill containing nested capabilities.

### Coding and self-debugging

The coding engine generates executable Python or Bash depending on task requirements. For scientific tasks, the engine is required to:

1. write runnable code;
2. execute it in a controlled environment;
3. inspect stdout, stderr and generated files;
4. write mini-tests for core functions when appropriate;
5. debug failed runs;
6. preserve scripts, figures and tables as artifacts.

For waveform analysis, code may use ObsPy, NumPy, SciPy, Matplotlib, Plotly or project-specific helper functions. For manuscript generation, the engine can compile LaTeX and attach citations.

### Hypothesis scoring

Hypothesis scores are updated iteratively. Each evidence record contributes support, contradiction or neutrality to one or more hypotheses. The score update can be implemented as a weighted heuristic or probabilistic model:

`score(H_i) = normalize(prior(H_i) + sum_j w_j * polarity(E_j, H_i) * confidence(E_j))`

The agent stops when either the maximum iteration count is reached or new evidence no longer changes hypothesis scores beyond a specified threshold.

### Case-study data

`[TO FILL]`

Describe:

- seismic network;
- waveform archive;
- catalog source;
- time range;
- magnitude range;
- phase-picking method;
- location method;
- uncertainty representation;
- auxiliary geological or operational data.

### Statistical analyses

`[TO FILL]`

Candidate analyses include:

- magnitude of completeness;
- b-value estimation and bootstrap uncertainty;
- event migration rate;
- nearest-neighbor clustering;
- waveform similarity;
- STA/LTA trigger diagnostics;
- phase residual analysis;
- location uncertainty propagation;
- cross-section geometry;
- comparison with injection, hydrological, stress or structural data.

### Literature comparison

`[TO FILL]`

The retrieved literature is used to determine whether the discovered question is novel, whether proposed mechanisms are physically plausible, and which alternative explanations should be tested.

### Manuscript generation

SAGE exports the final interpretation as a manuscript draft in Markdown and LaTeX. Figures and intermediate artifacts are inserted into Results or Supplementary Information according to their evidence role. Citations are attached to literature-derived claims.

---

## Figure Plan

### Figure 1. SAGE as an evidence-driven discovery system

Panel a: System architecture showing user question, retrieval, skills, coding engine, evidence table, hypothesis scoring and manuscript generation.

Panel b: Iterative loop from question decomposition to evidence retrieval, executable testing and hypothesis update.

Panel c: Example evidence record linking a claim to source, code and figure.

### Figure 2. From automated catalog to uncertainty-aware interpretation

Panel a: Deterministic catalog map.

Panel b: Location uncertainty clouds or bootstrap relocations.

Panel c: Difference between deterministic and uncertainty-aware geological interpretation.

Panel d: Sensitivity of key metric to uncertainty assumptions.

### Figure 3. Autonomous discovery of the focal scientific question

Panel a: Candidate questions generated by the agent.

Panel b: Evidence availability versus novelty matrix.

Panel c: Selected scientific question and competing hypotheses.

Panel d: Hypothesis score evolution through iterations.

### Figure 4. Executable tests discriminate mechanisms

Panel a: Event migration or spatiotemporal clustering.

Panel b: Magnitude-frequency or b-value analysis.

Panel c: Waveform similarity or STA/LTA trigger diagnostics.

Panel d: Summary evidence matrix for competing hypotheses.

### Figure 5. Manuscript construction from intermediate artifacts

Panel a: Evidence graph linking data, code, figures and claims.

Panel b: Generated figures and tables inserted into manuscript sections.

Panel c: Final traceable interpretation with evidence citations.

---

## Supplementary Information Plan

Supplementary Fig. 1: Full system prompt and agent loop.

Supplementary Fig. 2: Skill retrieval examples.

Supplementary Fig. 3: Code debugging trace.

Supplementary Fig. 4: Sensitivity tests for catalog thresholds.

Supplementary Fig. 5: Additional cross-sections and maps.

Supplementary Table 1: Evidence record table.

Supplementary Table 2: Hypothesis score updates.

Supplementary Table 3: Generated scripts and their outputs.

Supplementary Table 4: Literature retrieval summary.

---

## Candidate High-Impact Framing Options

### Option A: Method-first NC paper

**Core claim:** Evidence-driven scientific agents can make AI-assisted earthquake interpretation auditable and reproducible.

Best if the strongest novelty is the system architecture.

### Option B: Discovery-first NC paper

**Core claim:** SAGE discovers a new mechanism or unresolved question in a real earthquake sequence.

Best if the case study produces a genuinely new seismological finding.

### Option C: Catalog-uncertainty NC paper

**Core claim:** Uncertainty in automated catalogs is not noise; it changes the inferred physics of earthquake sequences.

Best if the paper builds on automated catalog uncertainty and robust location.

### Recommended route

The strongest NC route is **Option B + C**:

> Use SAGE as the discovery engine, but make the paper scientifically centered on a new earthquake-system finding revealed by uncertainty-aware automated catalogs.

In other words, the title should ultimately mention the discovered phenomenon, not only the AI system. SAGE becomes the engine that enabled the discovery.

Example final-title pattern:

**Uncertainty-aware automated catalogs reveal `[NEW MECHANISM]` in `[REGION]` earthquake sequences**

Subtitle or final paragraph:

**identified through an autonomous evidence-driven seismology agent**

---

## What SAGE Must Produce Before This Becomes Submission-Ready

1. A real case study dataset.
2. A final discovered scientific question.
3. At least two competing hypotheses.
4. A complete evidence table.
5. Reproducible code outputs.
6. At least four publication-quality figures.
7. A literature novelty analysis.
8. Statistical tests showing the finding is robust.
9. A manuscript LaTeX/PDF build.
10. Human expert review of all claims.

---

## Candidate Cover Letter Paragraph

Dear Editors,

We submit the manuscript “Autonomous evidence-driven seismology reveals testable earthquake-system hypotheses from data, literature and executable analysis” for consideration in Nature Communications. The study addresses a growing challenge in earthquake science: automated detection systems now produce catalogs at scales that exceed the capacity of conventional manual interpretation, yet the reasoning that connects these catalogs to geological conclusions remains difficult to audit. We introduce SAGE, an evidence-driven seismology agent that integrates retrieval, executable analysis, uncertainty-aware catalog interpretation, hypothesis scoring and manuscript generation. Applied to `[CASE STUDY]`, the system identifies `[KEY DISCOVERY]`, which we validate through `[KEY TESTS]`. The work provides both a new scientific finding and a general framework for trustworthy AI-assisted discovery in data-rich geoscience.

---

## Next Writing Step

To turn this draft into a real NC-style manuscript, the next step is to run evidence-geo-agent on one strong case study and fill:

- `[STUDY REGION]`
- `[NEW SCIENTIFIC QUESTION]`
- `[KEY FINDING]`
- `[HYPOTHESES]`
- `[FIGURES]`
- `[STATISTICAL RESULTS]`
- `[LITERATURE NOVELTY]`

After that, the manuscript should be rewritten around the discovered phenomenon rather than the software itself.

