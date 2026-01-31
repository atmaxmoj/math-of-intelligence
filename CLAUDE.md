# The Algebra of Intelligence

> **CRITICAL: DO NOT read PDF files directly!** PDFs are too large and will cause context overflow. Always use the pre-split `.txt` files in each PDF's corresponding folder (e.g., `01-basics/cerami-modal1-intro/01_Introduction.txt`). If txt files don't exist, run `scripts/.venv/bin/python scripts/pdf_split.py` first.

## Project Overview

This project explores a unified theory connecting **logic**, **learning**, and **structure**. The core vision:

> **Logic can finally learn.**

We're writing a book ("The Algebra of Intelligence") and conducting experiments that bridge:
- **Modal logic** as a language for describing structured systems
- **Coalgebra** as the categorical foundation for behavior and state
- **Semirings** as the algebraic structure enabling continuization and gradient descent
- **Learning theory** through the lens of distance matching and parameterization

## The Book Structure (`note/main.tex`)

Four parts:

| Part | Theme | Key Ideas |
|------|-------|-----------|
| **I: Expression** | Logic as language for describing systems | History → Kripke → Bisimulation → Completeness → Coalgebra |
| **II: Learning** | Making logic continuous/learnable | Semirings → Fuzzy logic → Gradient semiring → Differentiable structures |
| **III: Learning Well** | What makes learning work? | Distance matching → κ(J) → Coalgebra parameterization |
| **IV: Limits** | Boundaries of learning | Meta-learning → Fixed points → Uncomputability |

## Core Insights

### The Great Shift
Logic evolved from "studying valid arguments" to "describing structured systems." This explains why philosophers, linguists, and computer scientists all need logic—they all have systems to describe.

### The Hierarchy
```
Neural Semiring-valued Coalgebra  ← learn both structure AND operations
            ↓
Semiring-valued Coalgebra         ← learn structure, fixed semiring
            ↓
Crisp Coalgebra = Program         ← discrete, executable
```

### Distance Matching Principle
Good parameterization = parameter distance ≈ behavioral distance. See `experiments/RESULTS_SUMMARY.md` for empirical findings.

## Directory Structure

```
├── note/                    # LaTeX book
│   ├── main.tex            # Main file
│   ├── preamble.tex        # Packages and custom commands
│   └── chapters/           # Chapter files by part
├── 01-basics/ ... 07-handbook/   # Reading materials (PDFs + split txt)
├── ideas/                   # Research directions and references
├── experiments/             # Python experiments
│   ├── RESULTS_SUMMARY.md  # Key findings
│   └── *.py                # Experiment scripts
├── code/                    # Learning automata implementations
└── datasets/                # Benchmarks (automata, reasoning)
```

## LaTeX Compilation

MacTeX is installed at `/Library/TeX/texbin/`. To compile:

```bash
cd note
/Library/TeX/texbin/pdflatex main.tex
```

Or add to PATH: `export PATH="/Library/TeX/texbin:$PATH"`

## Writing Style

**Feynman lectures for logic.** Rigorous but human. Deep but accessible.

- Lecture-like tone: "we" and "let's", bring the reader along
- Simple language first, then notation
- Motivate before defining
- Visual variety: intuition boxes, warning boxes, diagrams, examples
- Historical context and "why do we care?"

Key LaTeX environments (defined in `preamble.tex`):
- `\begin{intuition}` - blue box with lightbulb
- `\begin{warning}` - red box for common pitfalls
- `\begin{keyinsight}` - green box for important observations
- `\begin{history}` - yellow box for historical notes
- `\begin{summary}` - purple box for chapter summaries

## Conversation Style

- **Chinese** for discussion, **English** for technical terms
- Be rigorous about definitions and proofs
- Challenge vague understanding—ask for precise statements
- Help develop intuition, not just memorize
- The user vibes and iterates; help structure and formalize

## Current Focus

The book's opening chapters (Part I):
1. **Chapter 1: Why Logic?** — History from Aristotle to category theory
2. **Chapter 2: Logic as Language** — The shift from consequence to structure description

Key themes to emphasize:
- The desire for greater expressive power runs through all of logic's history
- Logic shifted from "studying consequence" to "describing systems"
- Category theory (algebra/coalgebra) is the natural endpoint of this progression

## Research Directions

See `ideas/` folder for detailed notes:
- `coalgebra.md` - Coalgebra as the right abstraction
- `semiring.md` - Semiring as the right continuization
- `universal_learner.md` - Fixed point bypasses uncomputability
- `meta-learning.md` - Learning is coalgebraic, so meta-learning is too

## Ultimate Vision

A unified framework where:
- **Structure** is described by coalgebras
- **Language** is modal logic (the natural language of coalgebras)
- **Learning** is gradient descent on semiring-valued coalgebras
- **Intelligence** is the fixed point of meta-learning

The book forces clarity; the experiments test intuitions; the theory unifies.
