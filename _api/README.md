# API Documentation Directory

API reference documentation will go here. Each module should have its own page.

## Planned API Pages

- [ ] `statistics.md` - Statistics module (GLM, t-test, ANOVA, etc.)
- [ ] `machine-learning.md` - ML models and utilities
- [ ] `multivariate.md` - Ordination and clustering
- [ ] `plotting.md` - Visualization functions
- [ ] `preprocessing.md` - Data preprocessing utilities

## Template

```markdown
---
layout: default
title: "Module Name"
parent: API Reference
nav_order: 1
---

# Module Name

Brief description of the module.

---

## Classes

### ClassName

Description of the class.

**Constructor:**

\`\`\`javascript
new ds.module.ClassName(options)
\`\`\`

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Methods:**

#### `.fit(data)`

Description...

#### `.predict(data)`

Description...

**Example:**

\`\`\`javascript
const model = new ds.module.ClassName({ param: value });
model.fit({ data, X: features, y: target });
\`\`\`

---

## Functions

### functionName

Description...
```
