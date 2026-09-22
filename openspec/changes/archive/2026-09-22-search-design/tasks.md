## 1. Search contract and validation

- [x] 1.1 Confirm the search capability contract in the proposal and spec, and verify the change is scoped to observable behavior only.
- [x] 1.2 Validate the search design against the existing app and schema behavior, and confirm there are no unresolved design decisions that would change the contract.

## 2. Implementation follow-through

- [x] 2.1 Update the search parser and route logic required by the spec, and verify the targeted app tests for parsing and ranking pass.
- [x] 2.2 Adjust any safety checks or ranking parameters required by the contract, and verify the relevant search tests still pass.

## 3. Final verification

- [x] 3.1 Run the app-focused test suite for search behavior and confirm the change passes without regressions.
- [x] 3.2 Run `openspec validate --change "search-design"` and confirm the planning artifacts are complete and consistent.
