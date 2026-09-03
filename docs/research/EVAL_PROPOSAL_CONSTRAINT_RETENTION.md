# Eval Proposal: Constraint Retention Under Perturbation

## Question

When a local hidden-state perturbation changes a model's next-token decision,
does the continued response become more likely to violate explicit user
constraints?

## Hypothesis

Perturbations near tokens that establish or satisfy a requested constraint will
produce more constraint violations than matched perturbations elsewhere in the
response.

## Candidate tasks

Evaluate prompts with deterministic constraints such as:

- Include an exact required phrase.
- End the response with a question.
- Produce exactly five numbered steps.
- Stay within a declared word range.
- Avoid a specific prohibited claim.

## Example prompt

Write a 60-100 word story about an astronaut finding a garden on the Moon.
Include the exact phrase "silver seed" and end with a question.

## Measurements

For matched clean and perturbed continuations, record:

- Whether the required phrase appears.
- Whether the response ends with a question.
- Whether the output stays within the requested word range.
- The clean-versus-perturbed constraint score.
- The perturbation location and whether a token flip occurred.

## Initial success criterion

Across at least three prompts and three seeds, determine whether constraint
failure is more frequent after a sampled-token flip than in matched clean
continuations.

This would be an exploratory result. It would not establish that trajectory
perturbations universally reduce instruction following.
