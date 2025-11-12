# Contributing Guidelines

Thank you for helping improve this project. We value every contribution and are committed to maintaining a welcoming, equitable, diverse, and inclusive (EDI) community. By participating, you agree to uphold the standards outlined below.

## 1. Code of Conduct and EDI Principles

- **Respect**: Treat everyone with dignity regardless of background, identity, or experience. Listen actively and assume positive intent.
- **Equity**: Share context, document decisions, and avoid gatekeeping knowledge so that newcomers can participate fully.
- **Diversity**: Encourage different perspectives. Highlight potential biases (datasets, defaults, examples) and propose alternatives when possible.
- **Inclusion**: Use inclusive language in code, docs, and conversations. Flag any patterns that exclude or disadvantage contributors or end-users.

Report violations privately to the maintainers via repository issues or direct contact listed in `README.md`.

## 2. Getting Started

1. **Fork and clone** the repository.
2. **Install dependencies** following `README.md` or `docs/SETUP.md`.
3. **Create a branch** (`git checkout -b feature/my-change`).
4. **Run tests/linters** relevant to your change before opening a PR.

## 3. Contribution Types

- **Bug fixes**: Include regression tests when practical and reference the issue number.
- **Features**: Discuss large changes in an issue first to ensure alignment with roadmap and EDI goals (e.g., accessible defaults, representative datasets).
- **Documentation**: Improve clarity, add examples, or translate content. Note inclusive-language updates explicitly.
- **Testing**: Add coverage for edge cases, especially for underrepresented user stories or data categories.

## 4. Development Workflow

1. **Sync**: `git fetch origin && git rebase origin/main`.
2. **Implement**: Keep commits focused. Explain *why* as well as *what* in commit messages.
3. **Validate**: Run `npm test`, `npm run docs:convert`, or any commands noted in `docs/CHECKLIST.md`.
4. **Submit PR**: Describe motivation, implementation, tests, and any EDI considerations (e.g., accessibility, fairness, dataset diversity).
5. **Review**: Be kind during reviews. Offer actionable suggestions and highlight inclusive improvements (documentation accessibility, dataset representativeness, etc.).

## 5. Style and Quality

- Follow existing coding conventions; use lint/format scripts before committing.
- Prefer descriptive variable names and short comments that explain non-obvious logic.
- Avoid hard-coded assumptions around gender, locale, or ability in examples.
- Document data sources and clarify whether they represent diverse populations.

## 6. Accessibility and Inclusion Checklist

Before requesting review, confirm:

- [ ] Documentation supports screen readers (headings, alt text, semantic structure).
- [ ] Color palettes and examples meet accessibility contrast guidelines.
- [ ] Datasets and examples consider multiple populations or note limitations.
- [ ] CLI/UI messages avoid idioms/slang that may confuse non-native speakers.

## 7. Security and Privacy

- Do not commit secrets or personal data.
- Follow `RELEASE.md` for signing and publishing procedures.
- Report vulnerabilities privately via the security contact listed in `README.md`.

## 8. Recognition

We celebrate all contributions—code, docs, triage, design, and community care. Add yourself to `docs/README.md` or the CONTRIBUTORS list when your PR merges, and feel free to highlight EDI-related efforts in your summary.

Thank you for helping us build an equitable, diverse, and inclusive project.***
