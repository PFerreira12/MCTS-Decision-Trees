# Artificial Intelligence 2025/2026  
## Assignment — Adversarial Search Strategies and Decision Trees

## Overview

This project consists of developing two main Artificial Intelligence components:

1. A PopOut game-playing agent using Monte Carlo Tree Search (MCTS)
2. A Decision Tree implementation using the ID3 algorithm

The project combines:
- Adversarial Search
- Monte Carlo Tree Search
- Upper Confidence Bound for Trees (UCT)
- Decision Trees
- Entropy and Information Gain
- Dataset Generation
- Machine Learning fundamentals

---

# Project Goals

## Part 1 — PopOut + MCTS

Develop a fully functional PopOut game engine capable of supporting:

- Human vs Human
- Human vs Computer
- Computer vs Computer

The AI player must use:
- Monte Carlo Tree Search (MCTS)
- UCT evaluation strategy

The implementation should also explore:
- different MCTS configurations
- different node selection strategies
- performance analysis

---

## Part 2 — Decision Trees (ID3)

Implement a Decision Tree learner from scratch using:
- Entropy
- Information Gain
- ID3 recursive splitting

External ML libraries such as scikit-learn CANNOT be used for tree training.

The implementation must:
- train on datasets
- classify unseen examples
- visually represent the generated trees

---

## Part 3 — Dataset Generation

Generate a dataset from PopOut games played by the MCTS agent.

Each sample should contain:
- current board state
- best move selected by MCTS

This dataset will later be used to train a Decision Tree capable of approximating the MCTS behaviour.

---

# PopOut Game

## Game Rules

PopOut is a variation of Connect-4.

Players can:
- drop a piece into a column
- pop one of their own pieces from the bottom row

When a piece is popped:
- every piece above falls down one position

The winner is the first player to connect four pieces:
- horizontally
- vertically
- diagonally

---

## Special Rules

### Simultaneous Four-in-a-Row
If a pop move creates a win for both players:
- the player who made the pop move wins

### Full Board
If the board is full:
- the current player may either pop or declare a draw

### Repetition Rule
If the same game state occurs three times:
- either player may declare a draw

---

# Monte Carlo Tree Search (MCTS)

## MCTS Pipeline

### 1. Selection
Traverse the tree using UCT.

### 2. Expansion
Expand unexplored nodes.

### 3. Simulation
Simulate random playouts.

### 4. Backpropagation
Propagate results back through the tree.

---

# UCT Formula

```python
UCT = (wins / visits) + c * sqrt(log(parent_visits) / visits)
```

Where:
- exploitation measures node quality
- exploration encourages discovering new branches

---

# Suggested Experiments

## MCTS Variations

Test:
- different exploration constants
- different rollout strategies
- node pruning
- progressive widening
- move ordering heuristics

---

# Decision Trees (ID3)

## Required Features

Implement:
- entropy calculation
- information gain
- recursive tree construction
- prediction on unseen examples

---

# Entropy

```python
H(S) = - Σ p(x) log2 p(x)
```

---

# Information Gain

```python
IG(S,A) = H(S) - Σ (|Sv|/|S|) * H(Sv)
```

---

# Iris Dataset

The Iris dataset contains:
- sepal length
- sepal width
- petal length
- petal width

Target classes:
- Iris Setosa
- Iris Virginica
- Iris Versicolor

---

# Important Requirement — Discretization

Since the dataset contains numerical values:
- implement discretization manually
- select thresholds minimizing tree size

Possible approach:
```python
x < threshold
x >= threshold
```

Choose thresholds maximizing information gain.

---

# PopOut Dataset Generation

## Dataset Structure

Each sample:
```text
(state_i, move_i)
```

Where:
- state_i = current board configuration
- move_i = move chosen by MCTS

---

# Suggested Pipeline

## 1. Run Self-Play Matches
Generate many games using MCTS.

## 2. Save States and Moves
Store:
- board state
- selected action

## 3. Create Dataset
Export to CSV or dataframe.

## 4. Train ID3
Train a Decision Tree using generated data.

---

# Suggested Project Structure

```text
project/
│
├── README.md
├── requirements.txt
├── notebook/
│   └── project.ipynb
│
├── src/
│   ├── game/
│   │   ├── board.py
│   │   ├── rules.py
│   │   ├── moves.py
│   │   └── interface.py
│   │
│   ├── mcts/
│   │   ├── node.py
│   │   ├── mcts.py
│   │   ├── uct.py
│   │   └── simulation.py
│   │
│   ├── decision_tree/
│   │   ├── entropy.py
│   │   ├── information_gain.py
│   │   ├── id3.py
│   │   ├── discretization.py
│   │   └── predict.py
│   │
│   ├── dataset/
│   │   ├── iris_loader.py
│   │   ├── popout_generator.py
│   │   └── preprocessing.py
│   │
│   └── utils/
│       ├── visualization.py
│       └── metrics.py
│
├── data/
│   ├── iris.csv
│   └── popout_dataset.csv
│
├── reports/
│   ├── slides.pdf
│   └── auto_evaluation.pdf
│
└── results/
    ├── trees/
    ├── logs/
    └── plots/
```

---

# Technologies

## Recommended
- Python
- NumPy
- Pandas
- Matplotlib
- Graphviz

## Forbidden
- scikit-learn Decision Trees
- automatic ML tree generation libraries

---

# Development Plan

## Phase 1 — Game Engine
- Board representation
- Move generation
- Pop mechanics
- Win detection

---

## Phase 2 — MCTS
- Node structure
- Selection
- Expansion
- Simulation
- Backpropagation

---

## Phase 3 — Interface
- CLI interface
- Human interaction
- AI turns

---

## Phase 4 — ID3
- Entropy
- Information gain
- Recursive splitting
- Prediction

---

## Phase 5 — Dataset Generation
- Self-play
- Logging
- CSV export

---

## Phase 6 — Evaluation
- Accuracy
- Win rates
- Runtime analysis
- Tree complexity

---

# Evaluation Metrics

## MCTS
- win rate
- simulations per second
- average decision time

## Decision Tree
- classification accuracy
- tree depth
- node count

---

# Deliverables

## Mandatory Submission

### 1. Final Notebook
The notebook must:
- explain implementation decisions
- discuss experiments
- analyse results

### 2. Slides (PDF)
Maximum presentation time:
- 10 minutes

### 3. Auto-evaluation File

---

# Evaluation Criteria

| Component | Weight |
|---|---|
| MCTS Implementation | 30% |
| Decision Trees | 30% |
| Technical Quality | 30% |
| Communication Skills | 10% |

---

# Recommended Task Distribution

## Member 1
- Game engine
- Rules
- Interface

## Member 2
- MCTS
- Simulations
- Evaluation

## Member 3
- ID3
- Dataset generation
- Visualization

---

# Important Technical Notes

## Board Representation

Efficiency matters because MCTS generates many states.

Possible representation:
```python
tuple(board.flatten())
```

This can help:
- hashing
- repetition detection
- caching

---

# Final Objective

Build a complete AI pipeline where:
1. MCTS learns strong PopOut strategies
2. Self-play generates training data
3. ID3 learns from generated gameplay
4. The Decision Tree approximates MCTS decisions

---

# Main AI Concepts Used

- Adversarial Search
- Monte Carlo Tree Search
- UCT
- Decision Trees
- ID3
- Entropy
- Information Gain
- Self-play
- State-space search