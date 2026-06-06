import { describe, expect, it } from "vitest";
import { LettaRuleSolverProvider } from "../src/providers/adapters/LettaRuleSolverProvider.js";

describe("LettaRuleSolverProvider currentAction history handling", () => {
  it("appends currentAction when it appears earlier in history but is not last", async () => {
    const provider = new LettaRuleSolverProvider();
    provider.setRules("code_edit", [
      { type: "constrain_child_tools", tool_name: "read_file", children: ["grep"] },
      { type: "constrain_child_tools", tool_name: "grep", children: ["write_file"] },
    ]);

    const result = await provider.getAllowedNext(
      "code_edit",
      "read_file",
      ["read_file", "grep"],
      { availableActions: ["grep", "write_file"] },
    );

    expect(result.allowed).toEqual(["grep"]);
    expect(result.reason).toContain('Allowed after "read_file"');
  });

  it("does not duplicate currentAction when it is already the last history item", async () => {
    const provider = new LettaRuleSolverProvider();
    provider.setRules("code_edit", [
      { type: "constrain_child_tools", tool_name: "read_file", children: ["grep"] },
      { type: "max_count_per_step", tool_name: "read_file", max_count_limit: 1 },
    ]);

    const result = await provider.getAllowedNext(
      "code_edit",
      "read_file",
      ["read_file"],
      { availableActions: ["read_file", "grep"] },
    );

    expect(result.allowed).toEqual(["grep"]);
  });
});
