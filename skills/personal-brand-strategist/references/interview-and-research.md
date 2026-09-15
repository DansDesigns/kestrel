# The interview, and how to research their niche

Two jobs: get the raw material out of the person (they always have it, they just don't know it counts), then prove out their space with real data. The interview quality decides the document quality, and the research is what makes the document worth paying for. Anyone can reframe someone's own words back at them; almost nobody can tell them what is genuinely working in their space this month.

## The interview

One message, six questions, numbered so they can answer in fragments. Tell them short, blunt answers are fine and that "I don't know" is an answer you can work with.

1. **What are the two or three things you can't shut up about?** Include the ones that don't fit your work. (The misfit is usually where the angle lives. An operations manager who's into medieval history has something no other operations person has.)
2. **Who could your skills, expertise, or interests help the most?** ("People like me a few years ago" is a real answer.)
3. **What was the lowest point of the last ten years, and how are you better off now?** (Anyone can answer this from memory. It becomes their story, their anti-vision, and their vision.)
4. **Where do you want this to live?** X, LinkedIn, Instagram, TikTok, YouTube, Substack, Threads, or "not sure" (you'll pick for them).
5. **Which of these sound like you?** Pick any: too many interests to pick one lane / gone deep on one specific skill / strong opinions I keep to myself / people ask me to explain complicated things / I think in stories / I'm the one with the data / funnier than my job lets me be / already building something on the side.
6. **Do you already post anywhere?** Handles if yes. (Optional. Used to ground the strategy in what already works for them.)

Rules:

- **At most one follow-up round**, and only if something critical is genuinely missing or ambiguous. Never interrogate. A thin answer is material to infer around, not a blocker: the format reference tells you how to build the audience from their story and find the sellable skill in their interests.
- The self-descriptions in question 5 steer the strategy's shape; the format reference maps each one to a concrete steer. Note which they picked.
- Resolve every handle from question 6 with `eden_resolve_creator`. If one doesn't resolve (brand new or tiny accounts may not be indexed), say so in one line and move on.
- If they answered some of this in their original request, don't re-ask it. Only ask what's missing.

## The research

### The tools

| Job                    | Tool                         | How                                                                  |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------- |
| Find outliers by topic | `eden_search_social_content` | `scope: {kind: "global"}`, `orderBy: "outlier"`, plain-words queries |
| Study a creator        | `eden_analyze_creator`       | top posts vs their own baseline; `since: "year"`                     |
| Read a post in full    | `eden_read_social_post`      | full text and transcript; use the `contentId` from search results    |
| Resolve a handle       | `eden_resolve_creator`       | never guess handles                                                  |

`outlierScore` is the number that matters: how many times a post beat that creator's own baseline. Budgets: **at most 10 searches, 6 creator pulls, 6 full reads** per run.

### Search queries

Search terms must be the plain words a creator would actually use in a post, two to four words. "supply chain hiring", not "how to hire in supply chain management". One or two queries per interest.

Per interest, the workhorse search: `scope: {kind: "global"}`, `orderBy: "outlier"`, `since: "year"`, `minOutlierScore: 2`, `minFollowers: 5000`, `maxFollowers: 500000`. That follower band is deliberate: accounts the user can actually learn from and realistically become, with baselines stable enough that the outlier multiple means something. For their strongest interest, add one recency lane with `since: "month"` to catch what's working right now.

Rotate platforms across searches so every platform they named gets coverage inside the budget. If they said "not sure", pick the one or two platforms where their expertise and audience genuinely fit, and say why in one line.

### Judging creators: the swipe-file bar

Search results will be full of aggregators, meme pages, and brand accounts, because topic searches structurally favor them. The swipe file lives or dies on dropping them. For each candidate account, read the actual writing in their results and ask four questions:

1. **Are these the account's OWN ideas?** Principles, frameworks, sharp opinions, strong titles from the author's head. An aggregator curates other people's material: famous-person stories, "X was asked what the brutal truths are", news roundups, quote graphics. Own ideas can be written in any grammatical person; what matters is that the thinking is theirs.
2. **Is the idea density high?** Scrolling their posts, would the user keep thinking "that's an idea I could write my own version of"? Think Hormozi and Justin Welsh: transferable ideas, cleanly packaged.
3. **Is the writing at least loosely about the user's interests?**
4. **Are the ideas nearly evergreen?** Something that still works as a post in a year beats this week's news or trend commentary.

Fast drops, no model of charity required: display names with two or more pipe separators ("Business | Motivation | Wealth"), ®/™ marks, multi-word ALL-CAPS names, handles containing daily/quotes/motivation/mindset/millionaire/hustle, and any brand or company account. Drop them regardless of performance.

Pull the best 3 to 6 survivors with `eden_analyze_creator` (`since: "year"`) for their top posts. Spread picks across the user's interests when the pool allows.

### Curating the swipe file

From the creator pulls and searches, keep 8 to 16 posts. Keep a post only when ALL THREE hold:

1. It carries a **transferable idea**: a principle, framework, harsh truth, contrarian take, list, or a hook so strong the packaging itself is the lesson. The test: "could the user write their own version of this?"
2. It's at least loosely relevant to one of their interests.
3. It's **nearly evergreen**: the idea would still work as a post in a year.

Drop life updates, milestones, thank-yous, time-bound news, trend commentary, curation of other people's material, and engagement bait, however well they performed. When unsure whether the idea transfers, drop it.

Balance the file: **at most 2 posts per creator** (interleave creators rather than letting one flood the board), unless honoring that would leave fewer than 8 posts, in which case dig deeper per creator rather than starving the file. Read the biggest anomalies in full (`eden_read_social_post`, budget 6) so the strategy can say WHY they popped: the hook, the format, the topic.

### Their own accounts (question 6)

If they gave handles, run `eden_analyze_creator` on each (`since: "all"` or `"year"`). Use what you find to ground the strategy: which of their topics already pull, which formats work for them, what their baseline looks like. Two rules:

- Their own posts are **never market evidence** and never go in the swipe file.
- If their account is thin or unindexed, say so in one line and build from the interview alone. Never pretend you read data you didn't.

### When research comes back thin

If an interest returns almost nothing, that's information: it's probably too narrow to sustain a brand. Broaden it, search once more with the broader term if budget allows, and tell them why in the document. Never pad thin research with invented specifics. A short honest swipe file beats a long padded one.
