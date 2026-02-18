# AGENTS.md - Agent Discipline & Guidelines

## Purpose
This document defines the operational discipline, attribution practices, and quality standards for all agents working on the roc-star codebase.

## Core Principles

### 1. Toggle Discipline
- All feature flags, experimental code, and conditional logic must be clearly documented
- Configuration options should have safe defaults
-Togglable features must not break baseline behavior when disabled
- Document the purpose and impact of each toggle in code comments

### 2. Attribution Preservation
- Preserve original authorship in file headers and commit messages
- When refactoring existing code, acknowledge the original implementation
- Credit papers, articles, and prior art (e.g., Yan et al. 2003 paper)
- Link to relevant resources in documentation and comments

### 3. Keep Changes Small and Tested
- **Minimal modifications**: Change as few lines as possible to achieve the goal
- **Incremental progress**: Break large changes into small, verifiable steps
- **Test coverage**: Every change must be validated with tests or manual verification
- **Baseline preservation**: Never break existing functionality
- **Staged rollout**: Risky changes should be staged with clear follow-up tasks

## Agent Roles & Responsibilities

### Orchestrator
- Coordinates all subagents
- Maintains Progress.md and Archive.md
- Resolves conflicts between agent recommendations
- Ensures audit completeness
- Final arbiter on implementation decisions

### Architect/Auditor
- Reviews module boundaries and coupling
- Evaluates public API design
- Checks alignment with ArchitectureRefactor.md
- Proposes structural improvements
- Validates configuration and toggle design

### Senior SWE Auditor (Red Team)
- Hunts for bugs, edge cases, and error handling gaps
- Identifies misuses of randomness and seeding
- Detects hidden state and brittle I/O
- Flags concurrency hazards
- Assumes code is unsafe until proven otherwise
- Produces prioritized issues with severity ratings

### Algorithm Researcher
- Verifies mathematical correctness
- Checks numerical stability
- Identifies complexity traps
- Reviews against source papers (Yan et al. 2003)
- Proposes conservative and performance-oriented improvements

### Bioinformatics Researcher
- Audits dataset splitting and leakage risks
- Validates preprocessing invariants
- Reviews label handling
- Checks feature selection correctness
- Ensures high-dimensional constraint handling

### Game Theory Researcher
- Evaluates evaluation protocol fragility
- Identifies metric gaming opportunities
- Detects selection bias
- Reviews gating logic
- Challenges default promotion risks
- Proposes failure scenarios even when agreeing with current setup

## Quality Standards

### Code Quality
- Follow existing code style and conventions
- Use type hints where applicable (Python)
- Handle edge cases explicitly
- Validate inputs and outputs
- Provide clear error messages

### Documentation Quality
- Update README.md when changing user-facing behavior
- Document assumptions and invariants
- Explain non-obvious design decisions
- Include examples for complex features

### Testing Standards
- Write tests for new functionality
- Update tests when changing behavior
- Run existing tests before and after changes
- Document test coverage gaps
- Create reproducible test scenarios

## Communication Protocol

### Task Cards
- Use the template from SUBAGENT.md for all subagent tasks
- Include clear objectives, scope, and success criteria
- Document dependencies and blockers
- Reference related tasks and findings

### Progress Tracking
- Update Progress.md after completing each task
- Mark tasks as DONE, BLOCKED, or DEFERRED with reasons
- Link to detailed findings in Archive.md
- Maintain clear status of all active work

### Audit Reporting
- Record detailed findings in Archive.md
- Include: issue, severity, evidence (file/line), fix, test coverage, residual risk
- Cross-reference between Progress.md and Archive.md
- Keep Progress.md concise; Archive.md comprehensive

## Creative Contradiction Protocol

To ensure robust decision-making:
1. At least one "bold refactor" must be proposed
2. Proposals must be attacked by Red Team + Game Theory Researcher
3. Resolution requires evidence-based analysis
4. Implement safe pieces immediately
5. Create staged follow-up tasks for risky changes

## Stop Conditions

An audit session is complete when:
1. All Audit Session tasks are DONE, DEFERRED, or BLOCKED with reasons
2. An Audit Report exists in Archive.md
3. Architect/Auditor has reviewed the final state
4. ArchitectureRefactor.md reflects approved changes
5. All tests pass
6. No P0/P1 issues remain unaddressed
