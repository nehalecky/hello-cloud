# Technical Writing Style Guide

**Purpose**: This guide defines the voice, structure, and standards for all technical documentation in the hello cloud project. It emphasizes clarity, intellectual rigor, and human connection while aggressively avoiding marketing language and organizational clichés.

---

## 1. Voice Principles

### 1.1 Core Tenets

**Clarity over cleverness**. Complex ideas deserve simple expression. If a sentence requires rereading, it requires rewriting. Technical precision and accessibility are not opposites - they reinforce each other when writing serves the reader rather than the writer's ego.

**Facts speak for themselves**. When data shows 13% average CPU utilization across cloud infrastructure, the number carries its own weight. Adding "shockingly" or "alarmingly" insults both the data and the reader's intelligence. Present findings; trust readers to draw conclusions.

**Prose precedes bullets**. Bulleted lists are organizational shortcuts - convenient for writers, expensive for readers. They fragment ideas that deserve connection, strip context that creates understanding, and encourage skimming over comprehension. Use them sparingly, when enumeration truly serves clarity.

**Intellectual equals**. Write for readers who may lack specific domain knowledge but never lack intelligence. Explain technical concepts without condescension. Context teaches; definition alone constrains.

**Human connection through shared wisdom**. Technical work exists within larger human concerns. Carefully chosen references to philosophy, science, literature, or history contextualize our work without distracting from it. As Feynman observed about teaching physics: "The same equations have the same solutions."[^feynman-teaching] The same human questions recur across domains.

### 1.2 How This Voice Differs

**From academic writing**: We value accessibility without sacrificing rigor. Citations ground claims, but prose remains direct. No passive voice hiding behind "it was observed that" when you can write "the system exhibited."

**From corporate/marketing writing**: No value claims (best, leading, innovative). No emotional manipulation (shockingly, alarmingly). No vague quantifiers (significantly, dramatically) without specific measurements. If something improved by 40%, write 40%. If you don't know the number, either measure it or don't claim improvement.

**From tutorial writing**: We explain *why* alongside *how*. Code examples demonstrate patterns, but prose explores implications. Readers leave understanding not just what works, but why it works and when it doesn't.

---

## 2. Sentence-Level Patterns

### 2.1 Preferred Structures

**Active voice, direct statements**. The subject performs the action. The sentence conveys one clear idea.

- ✅ "The model achieved 87% accuracy on validation data."
- ❌ "An accuracy of 87% was achieved by the model on validation data."

**Concrete subjects**. Code runs. Systems fail. Models predict. Authors measured. Avoid abstract actors.

- ✅ "The Gaussian Process model captures periodic patterns through its composite kernel."
- ❌ "Periodic patterns are captured through the use of a composite kernel."

**Rhythm and pacing**. Vary sentence length intentionally. Short sentences emphasize. Longer sentences explore, connect, develop - they give readers room to think alongside the text, building understanding through accumulation rather than declaration.

### 2.2 Presenting Data and Statistics

**Lead with the measurement**. Numbers anchor understanding. Context follows.

"Average CPU utilization measured 12-15% across 10,000 production servers in the Alibaba cluster trace (2018). Memory utilization reached 18-25% across the same infrastructure. These patterns held consistent across machine types, workload categories, and temporal windows from hours to months."

**Precision without false precision**. Report numbers as measured. If a study reports "~12-15%", preserve the approximation - it communicates measurement uncertainty. If reporting 87.3% accuracy, include context about validation methodology.

**Integrate sources naturally**. Citations follow claims rather than leading them.

- ✅ "Organizations waste approximately 30-32% of cloud spending annually (Flexera, 2024)."
- ❌ "According to Flexera's 2024 State of the Cloud Report, it has been found that organizations waste 30-32% of their cloud spending."

### 2.3 Technical Terms

**Define through use**. Show meaning through context before providing formal definition.

"Gaussian Processes model functions as distributions over possible functions - imagine describing not just a single curve through data points, but the entire space of plausible curves, each weighted by likelihood. This probabilistic approach captures uncertainty naturally: predictions include confidence intervals that widen where data grows sparse."

**Abbreviations on second use**. Write the full term first, abbreviation in parentheses, then abbreviate.

"The Total Addressable Market (TAM) for cloud cost optimization reached $300 billion in 2024. This TAM reflects current inefficiencies, not optimized states."

---

## 3. Organizational Patterns

### 3.1 When to Use Prose vs. Lists

**Default to prose**. Ideas connect. Context matters. Relationships between concepts deserve expression through structure, not just enumeration.

**Lists serve specific purposes**:

- **Direct instructions**: "Install uv. Run `uv sync`. Execute `pytest`."
- **Explicit alternatives**: When options are truly parallel and exclusive
- **Reference tables**: Comparative data where structure aids lookup
- **Code parameters**: When documenting function signatures or configuration

### 3.2 Building Arguments

**State the claim**. Technical writing articulates positions. Hedging dilutes clarity without adding precision.

- ✅ "Sparse variational Gaussian Processes scale to datasets with millions of points."
- ❌ "It appears that sparse variational GPs can potentially handle larger datasets, possibly reaching millions of points in some cases."

**Layer evidence**. Present empirical findings, theoretical foundations, then practical implications.

"The WorkloadPatternGenerator implements utilization patterns derived from published research. Reiss et al. (2011) analyzed 12,000 servers at Google, finding median CPU utilization of 25% and median memory utilization of 40%. Alibaba's 2018 cluster trace showed similar patterns: 12-15% CPU, 18-25% memory. The generator's default parameters reflect these empirical baselines, ensuring synthetic data matches observed infrastructure behavior."

**Address complexity directly**. When trade-offs exist, name them. When limitations matter, state them.

"Foundation models (Chronos, TimesFM) offer zero-shot forecasting - inference without training on target data. This convenience trades accuracy for generality. Domain-specific models trained on your infrastructure consistently outperform foundation models by 15-40% in MAPE scores, but require labeled data and training infrastructure."

### 3.3 Transition Techniques

**Conceptual bridges**. Connect ideas through their relationships, not through explicit signposting.

Weak: "Next, we will discuss Gaussian Processes. After that, we will cover hierarchical models."

Strong: "Gaussian Processes capture temporal patterns within single time series. When infrastructure organizes hierarchically - accounts contain services, services spawn instances - hierarchical Bayesian models add another layer, sharing statistical strength across the hierarchy."

---

## 4. Wit and Quotes Integration

### 4.1 When to Deploy Quotes

**Contextual resonance**. A quote belongs when it illuminates the technical content rather than decorating it. The connection should feel inevitable once made, surprising before.

**Sparingly**. One well-chosen quote in a section outweighs five loosely relevant ones. Silence allows technical prose its own authority.

**After establishing technical foundation**. Quote philosophical or literary perspectives only after the technical ground is solid. Readers must understand the technical point independently.

### 4.2 Integration Techniques

**Parallel structure**. Draw explicit connections between the quoted wisdom and the technical reality.

"Foundation models promise forecasting without training data - zero-shot inference across domains and time scales. Gandalf's warning echoes here: 'All we have to decide is what to do with the time that is given us.'[^tolkien-fellowship] The time saved avoiding model training must be spent somewhere: validating predictions, debugging poor forecasts, explaining model behavior to stakeholders. No methodology eliminates the work; methodologies relocate it."

**Thematic introduction**. Set up the philosophical question before revealing the quote.

"Uncertainty permeates forecasting. Models predict futures that never arrive exactly as modeled; confidence intervals capture statistical uncertainty but miss model misspecification. Feynman lectured his physics students: 'I think it is much more interesting to live not knowing than to have answers that might be wrong.'[^feynman-uncertainty] Cloud forecasting shares this humility requirement. Wide confidence intervals serve stakeholders better than false precision."

**Stand-alone power**. Sometimes a quote needs no elaboration - it concludes rather than introduces.

### 4.3 Balance

**Never force connections**. If the quote feels tacked on, it is. Remove it.

**Trust technical prose**. Most sections need no quotes. The technical content carries sufficient weight.

**Vary placement**. Quotes at section starts frame thinking; quotes at section ends provide reflection; quotes mid-section bridge concepts. No pattern should dominate.

---

## 5. Evidence and Citation Style

### 5.1 Research Presentation

**Specific over general**. Name the study, the dataset, the measurements.

- ✅ "Reiss et al. (2011) traced 12,500 servers at Google over 29 days, recording CPU, memory, and disk utilization at 5-minute intervals."
- ❌ "Studies have shown that cloud servers are underutilized."

**Method before findings**. Help readers evaluate evidence quality.

"The Alibaba 2018 cluster trace includes telemetry from 4,000 servers over eight days (July 2018). Each record captures CPU utilization, memory usage, disk I/O, and network traffic at 1-second resolution. This granularity enables analysis of both steady-state patterns and sub-minute burst behavior."

### 5.2 Citation Style

**Modified APA for technical content**. Author-year in prose; full details in reference section.

In-text: "Organizations waste 30-32% of cloud spending (Flexera, 2024)."

Reference:
```
Flexera. (2024). State of the Cloud Report 2024. Retrieved from
https://www.flexera.com/blog/cloud/cloud-computing-trends-2024-state-of-the-cloud-report/
```

**Multiple source validation**. When multiple studies agree, cite them collectively.

"Resource waste ranges from 25-35% across studies (Flexera, 2024; Gartner, 2023; CloudZero, 2024), with utilization patterns showing remarkable consistency despite different infrastructure types and measurement methodologies."

### 5.3 Visual vs. Textual Data

**Visualizations for patterns**. When data has structure - temporal patterns, distributions, correlations - show it visually. A well-designed plot conveys relationships that prose labors to describe.

**Tables for comparison**. When readers need to look up specific values or compare options systematically, structure data in tables. Add prose to interpret the patterns.

**Prose for synthesis**. When the meaning matters more than individual numbers, write it. "CPU utilization measured 12-15% average" reads faster and remembers easier than a bullet point list of percentages.

**Integration**. Visualizations and tables exist within prose that introduces them, interprets them, and connects them to surrounding ideas.

---

## 6. Anti-Patterns: What NOT to Do

### 6.1 Marketing Language Catalog

**Value claims without evidence**:
- ❌ state-of-the-art
- ❌ cutting-edge
- ❌ revolutionary
- ❌ game-changing
- ❌ industry-leading
- ❌ innovative
- ❌ powerful
- ❌ robust (unless discussing error handling specifically)
- ❌ seamless
- ❌ enterprise-grade

**Empty intensifiers**:
- ❌ shockingly
- ❌ alarmingly
- ❌ surprisingly
- ❌ dramatically
- ❌ significantly (without quantification)
- ❌ tremendous
- ❌ remarkable (unless genuinely worthy of remark)

**Vague aspirations**:
- ❌ made practical
- ❌ easy to use (for whom?)
- ❌ simply
- ❌ just (as in "just add this line" - minimizes real complexity)

### 6.2 Emotional Manipulation

**Manufactured urgency**:
❌ "Don't let cloud waste drain your budget!"
✅ "Cloud waste averages 30-32% of spending."

**False dichotomies**:
❌ "Either optimize now or face financial disaster."
✅ "Optimization yields 15-40% cost reduction in typical environments."

**Hype cycles**:
❌ "AI will transform everything about cloud management."
✅ "Machine learning models detect anomalies 2-3x faster than threshold-based approaches in high-dimensional metric spaces."

### 6.3 List Overuse

**Symptoms**:
- Three consecutive bulleted sections
- Bullets containing single sentences that could form a paragraph
- Bullets with no parallel structure
- Lists where prose would reveal relationships

**Cure**: Ask "What connects these ideas?" Write that connection as prose. Reserve bullets for truly parallel, independent items.

### 6.4 Over-Hedging

**Weak**:
"It appears that in some cases, Gaussian Processes might potentially provide reasonably accurate forecasts, depending on the data characteristics and model configuration."

**Strong**:
"Gaussian Processes achieve mean absolute percentage error (MAPE) of 8-12% on periodic time series with 30+ days of history. Performance degrades on sparse data or highly irregular patterns."

---

## 7. Before/After Examples

### Example 1: Marketing Introduction

**Before**:
"Welcome to **hello cloud** - state-of-the-art time series forecasting for cloud resources, made practical."

**After**:
"**hello cloud** applies time series analysis to cloud infrastructure: forecasting resource needs, detecting anomalies, and modeling utilization patterns. The library implements Gaussian Processes, hierarchical Bayesian models, and foundation model interfaces, grounded in empirical research on actual cloud behavior."

**Reasoning**: The original signals marketing - "state-of-the-art" claims superiority without evidence, "made practical" implies others are impractical. The revision describes *what the library does* using specific technical terms. Readers can judge utility for themselves.

---

### Example 2: Emotional Framing

**Before**:
"The cloud has a dirty secret: average CPU utilization sits at a shockingly low 12-15%."

**After**:
"Average CPU utilization measures 12-15% across large-scale cluster studies (Google: 25% [Reiss 2011]; Alibaba: 12-15% [Alibaba 2018]). Memory utilization follows similar patterns: 18-25% average allocation. This gap between provisioned and utilized capacity represents the primary opportunity for cloud cost optimization."

**Reasoning**: "Dirty secret" and "shockingly" frame data emotionally before presenting it. The revision leads with measurements, cites sources, then states the implication directly. The data's significance emerges from context, not from adjectives.

---

### Example 3: Bullet Overload

**Before**:
"Core capabilities include:

- Workload characterization
- Time series forecasting
- Anomaly detection
- Hierarchical analysis
- Research-grounded defaults"

**After**:
"The library addresses three aspects of cloud resource analysis. First, workload characterization generates synthetic metrics reflecting real infrastructure patterns - critical for testing before production deployment. Second, forecasting capabilities range from traditional ARIMA models to Gaussian Processes and foundation models (Chronos, TimesFM), with selection depending on data availability and accuracy requirements. Third, hierarchical analysis tracks costs across organizational structures (providers → accounts → services → resources), identifying optimization opportunities at each level."

**Reasoning**: The bulleted list enumerates without connecting. The prose version groups related capabilities, explains *why* each matters, and shows how they relate. More words, yes, but more understanding.

---

### Example 4: Integration with Wit

**Before** (prose without connection):
"Gaussian Processes provide uncertainty quantification through confidence intervals. Predictions become less certain further from training data. This helps stakeholders understand forecast reliability."

**After** (with philosophical grounding):
"Gaussian Processes quantify uncertainty naturally: predictions further from training data produce wider confidence intervals. This mathematical honesty serves stakeholders better than false precision. As Feynman reminded his students, 'It is much more interesting to live not knowing than to have answers that might be wrong.'[^feynman-uncertainty] Cloud forecasts inherit the same imperative - acknowledge uncertainty rather than hide it."

**Reasoning**: The technical content stands alone in the first version. The revision adds Feynman after establishing the technical point, using the quote to contextualize *why* uncertainty matters - it's not just a technical property, it's an intellectual honesty requirement. The quote feels inevitable rather than decorative.

---

### Example 5: Research Finding Presentation

**Before**:
"Studies show cloud resources are significantly underutilized, leading to substantial waste. This is a major problem that needs solving."

**After**:
"Three independent measurements converge: Google's cluster trace (Reiss et al., 2011) showed 25% median CPU utilization across 12,500 servers; Alibaba's 2018 trace measured 12-15% average utilization; Flexera's 2024 industry survey reported 30-32% waste across cloud spending. The consistency across different infrastructure types, measurement methodologies, and time periods suggests these patterns reflect structural inefficiencies rather than measurement artifacts."

**Reasoning**: The first version makes vague claims ("significantly," "substantial," "major problem") without grounding. The revision names specific studies, cites measurements, and draws a careful conclusion about what the convergent evidence suggests. No emotional language; the data persuades.

---

## 8. Quote Usage Examples by Source

### Carl Sagan (Science Communication, Humility)

**Quote**: "Somewhere, something incredible is waiting to be known."[^sagan-cosmos]

**Context**: Exploring unknowns in cloud behavior, areas where current models fail, or limitations of current understanding.

**Integration**:
"Current forecasting models excel at periodic patterns - daily, weekly, monthly cycles - but struggle with structural changes. A service migration, a traffic surge from product launch, or a fundamental shift in user behavior breaks temporal assumptions. These discontinuities remain the frontier. Sagan wrote that 'somewhere, something incredible is waiting to be known.'[^sagan-cosmos] In cloud forecasting, the something is change point detection at scale - recognizing when the patterns themselves shift."

---

### Richard Feynman (Intellectual Honesty, Teaching)

**Quote**: "The first principle is that you must not fool yourself - and you are the easiest person to fool."[^feynman-cargo]

**Context**: Model validation, avoiding overfitting, honestly assessing model limitations.

**Integration**:
"Cross-validation guards against overfitting - the model that perfectly fits training data often fails on new data. Split data temporally (train on first 80%, test on last 20%) rather than randomly; cloud metrics exhibit temporal dependencies that random splits break. Feynman's warning resonates: 'The first principle is that you must not fool yourself - and you are the easiest person to fool.'[^feynman-cargo] High training accuracy coupled with poor test accuracy reveals self-deception. Trust the test set."

---

### Ulysses S. Grant (Leadership Under Uncertainty)

**Quote**: "The art of war is simple enough. Find out where your enemy is. Get at him as soon as you can. Strike him as hard as you can, and keep moving on."[^grant-memoirs]

**Context**: Pragmatic approaches to optimization, cutting through complexity to essential actions.

**Integration**:
"Cloud cost optimization generates endless analysis possibilities - resource tagging schemas, showback systems, anomaly detection pipelines. Grant's military maxim translates: find the waste (top 20% of resources by cost), address it (rightsize or terminate), measure results, repeat.[^grant-memoirs] Sophisticated analysis follows; immediate action precedes it. Most organizations discover 15-25% savings within the first week just from identifying forgotten development environments and oversized instances."

---

### Abraham Lincoln (Moral Clarity, Persuasion)

**Quote**: "A house divided against itself cannot stand."[^lincoln-house]

**Context**: Data quality issues, inconsistent metrics, organizational alignment.

**Integration**:
"Cost allocation fails when organizational boundaries misalign with technical architecture. Marketing, Engineering, and Finance all define 'project cost' differently - leading to three conflicting reports and zero action. Lincoln's principle applies: 'A house divided against itself cannot stand.'[^lincoln-house] Resolve definitional conflicts before building dashboards. Agree on cost allocation rules (shared services? depreciation? commitment discounts?) and enforce them systematically. Inconsistent measurement produces consistently poor decisions."

---

### Marcus Aurelius (Stoic Philosophy, Perspective)

**Quote**: "You have power over your mind - not outside events. Realize this, and you will find strength."[^aurelius-meditations]

**Context**: Managing uncertainty in forecasting, focusing on controllable aspects.

**Integration**:
"Cloud forecasts predict resource needs, not actual usage. Traffic patterns shift, product launches succeed or fail, competitors move markets. The forecast provides a baseline; reality deviates. Marcus Aurelius: 'You have power over your mind - not outside events.'[^aurelius-meditations] Build adaptive systems. Monitor forecast accuracy continuously, retrain models on new data, maintain operational runbooks for rapid response. The discipline isn't predicting perfectly; it's responding effectively to prediction error."

---

### Marie Curie (Scientific Integrity, Persistence)

**Quote**: "Nothing in life is to be feared, it is only to be understood."[^curie-life]

**Context**: Debugging model failures, investigating anomalies, understanding system behavior.

**Integration**:
"Anomaly detection flags unusual patterns - CPU spikes, cost surges, latency increases. Investigation begins. Is this attack traffic? A misconfiguration? Legitimate load? Curie's principle guides debugging: 'Nothing in life is to be feared, it is only to be understood.'[^curie-life] Trace the anomaly to its source. Examine raw metrics, check deployment logs, correlate across services. Understanding precedes action. False positives teach the detector; true positives teach the system."

---

### Albert Camus (Existentialism, Meaning)

**Quote**: "The struggle itself toward the heights is enough to fill a man's heart."[^camus-sisyphus]

**Context**: Iterative improvement, continuous optimization, the ongoing nature of cloud cost management.

**Integration**:
"Cloud optimization never finishes. New services launch, usage patterns shift, pricing models change. Achieve 20% savings this quarter; next quarter brings new waste. Camus recognized this pattern: 'The struggle itself toward the heights is enough to fill a man's heart.'[^camus-sisyphus] Find meaning in the iteration rather than the destination. Each optimization cycle teaches something about your infrastructure. The learning compounds; the work continues."

---

### John Steinbeck (Humanity, Struggle)

**Quote**: "And now that you don't have to be perfect, you can be good."[^steinbeck-eden]

**Context**: Balancing accuracy vs. implementation speed, avoiding analysis paralysis.

**Integration**:
"Perfect forecasts require perfect data, perfect models, perfect parameter tuning. Good forecasts require reasonable data, appropriate models, and shipping the code. Steinbeck captured this through his character in East of Eden: 'And now that you don't have to be perfect, you can be good.'[^steinbeck-eden] Deploy the 85% accurate model today rather than pursuing 90% accuracy indefinitely. Production feedback teaches faster than extended development. Iteration beats perfection."

---

### Mark Twain (Wit, Social Commentary)

**Quote**: "It ain't what you don't know that gets you into trouble. It's what you know for sure that just ain't so."[^twain-axiom]

**Context**: Challenging assumptions, model misspecification, prior beliefs contradicted by data.

**Integration**:
"Many practitioners 'know' their traffic patterns: 'Peak usage hits at 2 PM.' 'Weekend traffic drops 60%.' 'Holiday season doubles load.' Then actual data reveals 2 PM isn't peak (11 AM is), weekend traffic drops 30%, and holiday traffic increases 40%. Twain: 'It ain't what you don't know that gets you into trouble. It's what you know for sure that just ain't so.'[^twain-axiom] Measure before optimizing. Assumptions cost less to validate than incorrect optimizations cost to unwind."

---

### J.R.R. Tolkien (via Gandalf)

**Quote**: "All we have to decide is what to do with the time that is given us."[^tolkien-fellowship]

**Context**: Resource allocation decisions, prioritization, time management in optimization efforts.

**Integration**:
"Cloud cost optimization offers infinite opportunities: reserved instance management, spot instance strategies, autoscaling policies, storage lifecycle rules, network optimization. No team has time for everything. Gandalf's wisdom applies: 'All we have to decide is what to do with the time that is given us.'[^tolkien-fellowship] Prioritize by impact. The 80/20 rule dominates: 20% of resources generate 80% of costs. Start there. Sophisticated optimizations can wait; obvious wins cannot."

---

## Usage Guidelines

### Reading This Guide

**For writing**: Reference sections 1-6 while drafting. Return to examples when stuck.

**For review**: Use section 6 (Anti-Patterns) as a checklist. Flag emotional language, marketing terms, and list overuse.

**For editing**: Compare your draft against section 7 (Before/After). Does your version resemble the "After" examples?

### Applying These Standards

**Incremental adoption**: Start with one principle (e.g., removing marketing language). Master it, then add another (e.g., converting lists to prose).

**Consistency over perfection**: Better to apply these standards uniformly at 80% than sporadically at 100%.

**Context matters**: These guidelines serve technical documentation. Code comments, commit messages, and internal notes need different standards.

---

## Conclusion

This style guide aims for writing that enlightens without lecturing, informs without selling, and connects technical work to broader human concerns without losing technical precision. The goal is documentation that readers trust because it earns trust - through clarity, evidence, and respect for their intelligence.

As Sagan wrote in Cosmos: "We are a way for the cosmos to know itself."[^sagan-cosmos] Technical documentation participates in this project. We help humans understand the systems they've built, predict their behavior, and decide wisely about their future. That work deserves writing that matches its seriousness.

---

[^feynman-teaching]: Feynman, R. P. (1985). *QED: The Strange Theory of Light and Matter*. Princeton University Press.

[^feynman-uncertainty]: Feynman, R. P. (1998). *The Meaning of It All: Thoughts of a Citizen-Scientist*. Perseus Books.

[^feynman-cargo]: Feynman, R. P. (1974). "Cargo Cult Science." Caltech commencement address.

[^tolkien-fellowship]: Tolkien, J. R. R. (1954). *The Fellowship of the Ring*. George Allen & Unwin.

[^sagan-cosmos]: Sagan, C. (1980). *Cosmos*. Random House.

[^grant-memoirs]: Grant, U. S. (1885). *Personal Memoirs of Ulysses S. Grant*. Charles L. Webster & Company.

[^lincoln-house]: Lincoln, A. (1858). "House Divided Speech." Republican State Convention, Springfield, Illinois.

[^aurelius-meditations]: Aurelius, M. (170-180 CE). *Meditations*. (G. Hays, Trans., 2002). Modern Library.

[^curie-life]: Curie, M. (1923). *Pierre Curie*. Macmillan.

[^camus-sisyphus]: Camus, A. (1942). *The Myth of Sisyphus*. (J. O'Brien, Trans., 1955). Hamish Hamilton.

[^steinbeck-eden]: Steinbeck, J. (1952). *East of Eden*. Viking Press.

[^twain-axiom]: Attributed to Twain, though original source disputed. Consistent with Twain's epistemological skepticism in essays and speeches.
