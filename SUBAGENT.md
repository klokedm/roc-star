# SUBAGENT.md - Task Card Template

## Purpose
This document provides the standard template for spawning and tracking subagent tasks in the TABNETICS audit workflow.

## Task Card Template

```markdown
### SUBAGENT TASK CARD: [TASK_ID]

**Agent Role**: [Architect/Auditor | Senior SWE | Algorithm Researcher | Bioinformatics Researcher | Game Theory Researcher]

**Task ID**: [Unique identifier, e.g., AUD-001, SWE-003, ALG-002]

**Priority**: [P0-Critical | P1-High | P2-Medium | P3-Low]

**Objective**: 
[Clear, concise statement of what this task aims to achieve]

**Scope**:
- Files/modules to review: [List specific files or modules]
- Focus areas: [Specific aspects to examine]
- Out of scope: [What NOT to include]

**Context**:
[Background information, relevant papers, prior discussions]

**Success Criteria**:
- [ ] [Specific, measurable outcome 1]
- [ ] [Specific, measurable outcome 2]
- [ ] [Specific, measurable outcome 3]

**Deliverables**:
- [ ] Findings document (section in Archive.md)
- [ ] Recommended actions (with priorities)
- [ ] Test coverage plan (if applicable)
- [ ] Updated documentation (if applicable)

**Dependencies**:
- Depends on: [Other task IDs that must complete first]
- Blocks: [Task IDs that are waiting on this]

**Timeline**:
- Expected effort: [estimate in hours/days]
- Target completion: [date or milestone]

**Status**: [NOT_STARTED | IN_PROGRESS | BLOCKED | UNDER_REVIEW | DONE]

**Assigned to**: [Agent role or specific agent]

**Notes**:
[Additional context, constraints, or considerations]
```

## Example Task Cards

### Example 1: Architecture Review

```markdown
### SUBAGENT TASK CARD: ARCH-001

**Agent Role**: Architect/Auditor

**Task ID**: ARCH-001

**Priority**: P1-High

**Objective**: 
Evaluate module boundaries and public API design for the roc-star loss function implementation.

**Scope**:
- Files/modules to review: rocstar.py, example.py
- Focus areas: Function signatures, parameter validation, public vs private APIs
- Out of scope: Algorithm correctness (ALG-002), numerical stability (ALG-003)

**Context**:
Initial architecture review to establish baseline and identify coupling issues.

**Success Criteria**:
- [ ] Public API documented with clear contracts
- [ ] Parameter validation identified
- [ ] Module boundary violations flagged
- [ ] Configuration toggle opportunities identified

**Deliverables**:
- [ ] Architecture Findings section in Archive.md
- [ ] Required refactor tasks list
- [ ] API improvement recommendations

**Dependencies**:
- Depends on: None (first task)
- Blocks: ARCH-002 (refactoring tasks)

**Timeline**:
- Expected effort: 2-3 hours
- Target completion: Session Day 1

**Status**: NOT_STARTED

**Assigned to**: Architect/Auditor subagent

**Notes**:
Focus on alignment with ArchitectureRefactor.md principles once that document is created.
```

### Example 2: Bug Hunting

```markdown
### SUBAGENT TASK CARD: SWE-001

**Agent Role**: Senior SWE Auditor (Red Team)

**Task ID**: SWE-001

**Priority**: P0-Critical

**Objective**: 
Identify critical bugs, edge cases, and error handling gaps in the loss function implementation.

**Scope**:
- Files/modules to review: rocstar.py, example.py
- Focus areas: Edge cases, error handling, randomness usage, state management
- Out of scope: Performance optimization (unless it impacts correctness)

**Context**:
Red team security and correctness review. Assume code is unsafe until proven otherwise.

**Success Criteria**:
- [ ] All edge cases documented (empty batches, single class, extreme values)
- [ ] Error handling reviewed
- [ ] Random seeding issues identified
- [ ] State management hazards flagged
- [ ] Concurrency safety evaluated

**Deliverables**:
- [ ] Prioritized issue list with severity ratings
- [ ] Concrete fix proposals for each issue
- [ ] Test cases to validate fixes

**Dependencies**:
- Depends on: None (can run in parallel with ARCH-001)
- Blocks: SWE-002 (bug fix implementation)

**Timeline**:
- Expected effort: 3-4 hours
- Target completion: Session Day 1

**Status**: NOT_STARTED

**Assigned to**: Senior SWE Auditor subagent

**Notes**:
Pay special attention to torch.rand_like() usage and GPU memory assumptions.
```

## Usage Guidelines

1. **Copy the template** when creating a new subagent task
2. **Fill all sections** - don't leave placeholders
3. **Be specific** in objectives and success criteria
4. **Update status** regularly in Progress.md
5. **Link to Archive.md** for detailed findings
6. **Cross-reference** related tasks using Task IDs
7. **Update dependencies** when they change

## Status Definitions

- **NOT_STARTED**: Task defined but work not begun
- **IN_PROGRESS**: Actively being worked on
- **BLOCKED**: Cannot proceed due to dependency or external factor
- **UNDER_REVIEW**: Work complete, awaiting review/approval
- **DONE**: Completed and validated

## Priority Definitions

- **P0-Critical**: Blocks progress, security issue, or data corruption risk
- **P1-High**: Significant impact on correctness or maintainability
- **P2-Medium**: Improvement opportunity, moderate impact
- **P3-Low**: Nice-to-have, minimal impact
