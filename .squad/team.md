# Squad Team

> onnx-shape-inference

## Coordinator

| Name | Role | Notes |
|------|------|-------|
| Squad | Coordinator | Routes work, enforces handoffs and reviewer gates. |

## Members

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Ripley | 🏗️ Lead / Shape-Inference Architect | .squad/agents/ripley/charter.md | active |
| Dallas | 🔧 Op Implementation Dev | .squad/agents/dallas/charter.md | active |
| Lambert | 🧪 Test & QA Engineer | .squad/agents/lambert/charter.md | active |
| Parker | 📝 Docs & DevRel | .squad/agents/parker/charter.md | active |

## Built-ins

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Scribe | 📋 Session Logger | .squad/agents/scribe/charter.md | active |
| Ralph | 🔄 Work Monitor | .squad/agents/ralph/charter.md | active |
| Rai | 🛡️ RAI Reviewer | .squad/agents/Rai/charter.md | active |
| Fact Checker | 🔍 Verifier | .squad/agents/fact-checker/charter.md | active |


## Coding Agent

<!-- copilot-auto-assign: false -->

| Name | Role | Charter | Status |
|------|------|---------|--------|
| @copilot | Coding Agent | — | 🤖 Coding Agent |

### Capabilities

**🟢 Good fit — auto-route when enabled:**
- Bug fixes with clear reproduction steps
- Test coverage (adding missing tests, fixing flaky tests)
- Lint/format fixes and code style cleanup
- Dependency updates and version bumps
- Small isolated features with clear specs
- Boilerplate/scaffolding generation
- Documentation fixes and README updates

**🟡 Needs review — route to @copilot but flag for squad member PR review:**
- Medium features with clear specs and acceptance criteria
- Refactoring with existing test coverage
- API endpoint additions following established patterns
- Migration scripts with well-defined schemas

**🔴 Not suitable — route to squad member instead:**
- Architecture decisions and system design
- Multi-system integration requiring coordination
- Ambiguous requirements needing clarification
- Security-critical changes (auth, encryption, access control)
- Performance-critical paths requiring benchmarking
- Changes requiring cross-team discussion

## Project Context

- **Project:** onnx-shape-inference
- **Created:** 2026-07-22
