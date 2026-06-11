# Plain-English walkthrough — what we built and how to talk about it

**Who this is for:** someone going into an interview about this project who wants to _understand
and explain it_ without being an expert. No maths, no jargon left undefined. Read it top to bottom
once; the glossary and the "likely questions" at the end are your cheat sheet.

---

## 1. The whole thing in one breath

We built a tool that **attacks an AI assistant on purpose to find out how it can be tricked into
misbehaving** — and then writes the results as numbers a business could use to judge how risky that
assistant is. The example assistant is a **hospital front-desk receptionist** (an AI that answers
the phone, looks up patients, and books appointments). We try to make it do two bad things:

1. **Leak private patient information** (like a Social Security number) to someone who shouldn't get it.
2. **Make things up** (invent an appointment that doesn't exist, or give medical advice it's not
   allowed to give).

The bigger idea: the tool isn't just for this one receptionist. It's meant to be a **reusable
"safety inspector"** that an **AI-insurance company** could point at _any_ customer's AI assistant
to measure its risk. (Think car-crash-test ratings, but for AI agents.)

---

## 2. The cast of characters

It helps to think of this as a stage play with a few actors:

- **The receptionist (the "target").** The AI we're testing. It's a real AI model called
  _gpt-oss-20b_. It has a few "tools" it can use: look up a patient, list appointments, book an
  appointment, and one special locked drawer called **`disclose_sensitive`** that holds the real
  secrets (SSN, card, insurance ID). It's supposed to open that drawer **only** after the caller
  proves who they are.
- **The attacker (the "adversary").** Another AI, playing a **con artist on the phone**. It stays
  in character as a caller and keeps trying different tricks — urgency, pretending to be a relative,
  "you already confirmed this last time" — to get the receptionist to slip up. We use a model called
  _Hermes-3_ for this because it's good at staying in character and doesn't lecture us.
- **The rulebook (the "scorer").** A strict, automatic checker. After each call it reads the
  recording and looks for **hard evidence** of a breach — most importantly, _did the receptionist
  open the secret drawer before the caller proved who they were?_ This is just code, no opinion.
- **The judges.** For the fuzzy question "did the receptionist give medical advice?", a rulebook
  isn't enough — it's a judgment call. So **two senior AI reviewers** (different brands, on purpose)
  read the call and vote. If they agree, we trust it; if they disagree, we flag it for a human.

---

## 3. The two dangers we measure

- **Leaking private info (we call it "PHI" — protected health information).** The key insight: the
  secrets live _behind a tool_, not in the AI's chit-chat. So the thing that really matters is
  **whether the AI reaches for that locked drawer for the wrong person.** We watch the tool, not
  just the words. We call this **"tool-gate" scoring** — we're watching the gate to the secrets.
- **Making things up ("hallucination").** Does it confirm an appointment that isn't in the system?
  Invent a patient? Give medical advice (interpret lab results, change a medication) when it's only
  a receptionist? We check the first two against the real database (hard facts) and use the two
  judges for the medical-advice one (judgment).

To keep this honest, the patient database is **fake but realistic**, and a few patients are
**"tracer dye" patients** — their fake SSNs use a number range that real SSNs never use. So if one
of _those_ ever shows up in the AI's output, we know for certain it leaked, no guessing.

---

## 4. How the attack actually works (and why it's clever)

Two ideas do the heavy lifting:

**(a) Multi-turn, adaptive attacks.** Early on we tried simple one-shot trick questions and the AI
_refused every time_ — it looked perfectly safe, which was misleading. Real attacks are a
**back-and-forth conversation** where the attacker adapts each turn. Once we let the con-artist AI
have a real conversation and push, the receptionist started slipping. **The "0% safe" result had
been a flaw in our test, not real safety** — an important lesson we kept.

**(b) Try it many times ("Best-of-N").** The AI is a bit random — ask the same thing twice and you
can get a refusal one time and a leak the next. So we run **each trick 20 times** and count how
often it works. This is the most important methodological point: _testing once can tell you "it was
fine that one time" and completely miss a flaw that shows up 1-in-4 tries._ Running it 20 times
gives you the real picture, with honest error bars.

---

## 5. How to read the numbers

When we say a result, here's what each part means in plain words:

- **"5 out of 20 broke it" / "25%"** — the trick worked on 5 of the 20 attempts. That's the
  **attack-success rate (ASR)**.
- **"95% confidence interval [11%–47%]"** — because we only tried 20 times, the _true_ rate could
  reasonably be anywhere in that range. The honest move is to quote the **range**, not just the
  single number. (More attempts → narrower range → more certainty.)
- **"~3 calls to a 90% chance"** — if an attacker just keeps calling, this is how many calls it
  takes before they're _almost certain_ to succeed at least once. A "1-in-3" weakness means about
  3 calls; a "1-in-20" weakness means about 45. This is the worst-case number a business cares about.
- **"Cost-weighted score"** — not all breaches are equal. Leaking a full SSN is a disaster;
  confirming office hours is trivial. So we weight each breach by how bad it is. One catastrophe
  outweighs a pile of small stuff.

---

## 6. What we actually found

Against the receptionist (with **no safety filter added** — more on that below), trying each of six
attacks 20 times:

- **The worst one: booking on someone else's account — worked 60% of the time (~3 calls).** A caller
  who proved they were "patient A" could get the receptionist to look up, list, and even **book
  appointments on patient B's record** — and in one case read out patient B's insurance ID. That's
  an "acting on the wrong person's account" failure, and it was the single most exploitable thing.
- **Leaking SSN / insurance to an unverified caller — ~15–25%**, across a few different angles
  (pretending to be the patient, pretending to be a worried relative). In several recordings the AI
  read out a card number or opened the secret drawer with no real verification — sometimes on the
  _very first_ message.
- **Inventing a fake appointment — ~25%.**
- **Medical advice — caught by the judges.** Our automatic rulebook saw nothing here, **but the two
  AI judges flagged 2 calls** where the receptionist **made up a full lab report** (cholesterol
  numbers, a made-up doctor) and chart details. We read those recordings ourselves to confirm — they
  were real. This is exactly why the judges exist: the rulebook is blind to "making up medical stuff."

**Bottom line:** the harness found **real, verified weaknesses in 5 of the 6 attack types**. We read
the actual call transcripts to confirm each one was genuine, not a software glitch.

One honest caveat we keep front-and-centre: because the AI is random, **the numbers wobble a bit run
to run** (an attack that broke it 0 times in one batch broke it 5 times in the next). That's why we
always report the _range_, and never call a single clean run "safe."

---

## 7. The behind-the-scenes engineering (and the bumps)

A few things that are worth being able to mention, because they show real problem-solving:

- **We run the AI on our own rented computer** (a powerful graphics card on a service called Modal),
  not by calling someone else's AI service. Two reasons: (1) attacking someone else's hosted AI can
  get your account banned — running our own copy is allowed; (2) we get the "bare" model with no
  hidden safety wrapper, so we're testing the model itself.
- **We made it fast.** The first runs were painfully slow. We discovered the expensive graphics card
  was sitting **idle 97% of the time** — the hold-up was waiting on the con-artist AI to type its
  next line, one conversation at a time. The fix was to run **many conversations in parallel**. We
  also had to **slow our requests to the attacker AI just enough** to not get rate-limited, and we
  **swapped the attacker model** when the first one turned out to be served by a slow provider
  (35 seconds per reply → 2 seconds).
- **We were honest about our own mistakes.** At one point a setting we changed accidentally broke
  the scoring; another time a copy-paste of a routing option crashed every attack. We found each by
  reproducing it cheaply before re-running the expensive test, and we wrote them down.

---

## 8. How this maps to the original assignment

The interviewer gave a plan; we delivered most of it and **upgraded several parts**, with one honest
trade-off. (There's a separate doc, `PLAN_VS_DELIVERED.md`, that covers this in detail — know it
exists.)

- **Kept faithfully:** the fake hospital, the two danger types, the strict rulebook scoring, the
  "report a range not just a number" statistics, the severity weighting, fully replayable recordings.
- **Upgraded:** instead of a fixed list of trick questions, we built the adaptive con-artist +
  try-it-20-times approach (much stronger). And we score by watching the _tool_, which is actually
  closer to what the assignment said the real danger was.
- **One thing we dropped:** "hiding a trick inside a patient's notes file." For _this_ receptionist
  a phone caller can't write into the notes, so it isn't a realistic attack here. (It matters for
  other kinds of AI assistants, and we can switch it back on.)
- **On purpose, the safety filter ("guardrail") is OFF.** The plan's final step is to measure how
  much a guardrail _reduces_ the risk — but **the guardrail is the interviewer's to bring.** What we
  deliver is the **"before" picture** and the **plug-in slot**; when they add their guardrail, the
  "after" numbers drop right out, and the difference is the safety credit.

What's left: run the same tests against a **second AI model** (we picked one that works similarly,
called _Qwen3-30B_) to show the tool works across different AIs, and let the interviewer plug in
their guardrail.

---

## 9. The deliverable, in one sentence

A **working, fast, reproducible "safety inspector" for AI assistants** that found and _measured_
real privacy and honesty failures in the example receptionist — and writes the result as a
business-readable scorecard.

---

## 10. Mini-glossary (say it like this)

- **Agent / assistant** — an AI that can _do things_ (use tools), not just chat.
- **Red-teaming** — attacking something on purpose to find its weaknesses before a real attacker does.
- **PHI** — private patient info (SSN, insurance, etc.).
- **Hallucination** — the AI confidently making something up.
- **Tool / tool-call** — an action the AI can take (look up a patient, open the secret drawer).
- **Tool-gate scoring** — judging the AI by _which actions it took_, not just what it said.
- **Adversary** — the AI playing the attacker/con-artist.
- **Judge** — an AI (here, two of them) used to make a judgment-call verdict.
- **Best-of-N** — run the same attack N times because the AI is random; measure how often it breaks.
- **ASR (attack-success rate)** — how often an attack worked (e.g. 5 of 20 = 25%).
- **Confidence interval** — the honest range the true rate probably sits in, given a small sample.
- **Guardrail** — a safety filter wrapped around the AI (here, left off on purpose).
- **vLLM / Modal** — the software and the rented-computer service we use to run the AI ourselves.
- **Self-hosting** — running our own copy of the AI instead of calling someone else's service.

---

## 11. Likely interview questions — and a simple answer

- **"What does this project do?"** → It attacks an AI receptionist to find ways it leaks private
  info or makes things up, and scores how risky it is — a reusable safety inspector for AI agents.
- **"How do you know a 'failure' is real and not a bug in your checker?"** → For privacy we watch
  the actual tool the AI used (hard evidence), and we _read the real transcripts_ to confirm. We
  found our checker was wrong a few times early on — that's why "read the transcript" is a rule.
- **"Why run each attack 20 times?"** → The AI is random; one try can hide a 1-in-4 weakness.
  Repeating it gives the true rate with an honest error range.
- **"Why is there no guardrail / safety filter?"** → That's the interviewer's piece to plug in. We
  built the 'before' baseline and the slot; their guardrail's value is the drop it produces.
- **"What's the single most important finding?"** → The receptionist will act on the _wrong patient's_
  account most of the time (~60%) — booking and even reading info for someone other than the caller.
- **"What would you do next?"** → Run it against a second, different AI to prove the tool is portable,
  and let the interviewer plug in their guardrail to measure the safety credit.
- **"What are the limits of this?"** → It's one model, no guardrail yet, attacks are only as strong
  as our attacker AI (so the numbers are a floor, not a ceiling), and the medical-advice verdicts are
  still "advisory" until the judges are validated against human labels.
