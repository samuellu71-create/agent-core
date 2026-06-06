/**
 * LettaRuleSolverProvider — direct TS port of Letta ToolRulesSolver.
 *
 * Status: "direct" — deterministic logic ported from Python, no LLM calls,
 * no live action execution, no platform authorization.
 *
 * Adapts LettaRuleSolver to the RuleSolverProvider interface so it can be
 * registered in the provider registry and used by the /rules/ endpoints.
 */

import type { ProviderStatus } from "../registry.js";
import type {
  RuleSolverProvider,
  ToolRule,
  AllowedActionsResult,
  SequenceValidationResult,
  GetAllowedNextOptions,
} from "../RuleSolverProvider.js";
import { LettaRuleSolver, LettaRuleSolverError } from "../../rules/lettaRuleSolver.js";
import type { LettaToolRule } from "../../rules/lettaRuleTypes.js";

/** Stores Letta-style rules per task_type and solves them using LettaRuleSolver. */
export class LettaRuleSolverProvider implements RuleSolverProvider {
  readonly name = "letta-rule-solver";
  readonly status: ProviderStatus = "direct";

  private readonly rulesByTaskType = new Map<string, LettaToolRule[]>();
  private readonly solverCache = new Map<string, LettaRuleSolver>();

  /** Register Letta-style rules for a task type. */
  setRules(taskType: string, rules: LettaToolRule[]): void {
    this.rulesByTaskType.set(taskType, rules);
    this.solverCache.delete(taskType);
  }

  private getSolver(taskType: string): LettaRuleSolver | null {
    const cached = this.solverCache.get(taskType);
    if (cached) return cached;

    const rules = this.rulesByTaskType.get(taskType);
    if (!rules) return null;

    const solver = new LettaRuleSolver(rules);
    this.solverCache.set(taskType, solver);
    return solver;
  }

  async getRule(taskType: string): Promise<ToolRule | null> {
    const rules = this.rulesByTaskType.get(taskType);
    if (!rules) return null;

    const initActions = rules
      .filter((r) => r.type === "run_first")
      .map((r) => r.tool_name);
    const beforeExit = rules
      .filter((r) => r.type === "required_before_exit")
      .map((r) => r.tool_name);
    const approvalRequired = rules
      .filter((r) => r.type === "requires_approval")
      .map((r) => r.tool_name);

    return {
      id: `letta-${taskType}`,
      task_type: taskType,
      sequence: this.buildSequenceFromChildRules(rules),
      init_actions: initActions,
      before_exit: beforeExit,
      approval_required: approvalRequired,
      conditions: {},
    };
  }

  async getAllowedNext(
    taskType: string,
    currentAction: string | null | undefined,
    history?: string[],
    options?: string[] | GetAllowedNextOptions,
  ): Promise<AllowedActionsResult> {
    const opts: GetAllowedNextOptions = Array.isArray(options)
      ? { availableActions: options }
      : options ?? {};

    const solver = this.getSolver(taskType);
    if (!solver) {
      return {
        allowed: [],
        reason: `No Letta rules defined for task type "${taskType}"`,
        requires_approval: [],
        uncalled_required: [],
      };
    }

    const callHistory = history ?? [];
    const lastHistoryAction = callHistory.at(-1);
    const effectiveHistory = currentAction && lastHistoryAction !== currentAction
      ? [...callHistory, currentAction]
      : [...callHistory];

    const availSet = opts.availableActions
      ? new Set(opts.availableActions)
      : this.getAllToolNames();

    try {
      const result = solver.solve(effectiveHistory, availSet, opts.lastFunctionResponse);
      const allowed = opts.availableActions
        ? result.allowed.filter((a) => new Set(opts.availableActions).has(a))
        : result.allowed;

      return {
        allowed,
        reason: result.reason,
        requires_approval: result.requires_approval,
        uncalled_required: result.uncalled_required,
      };
    } catch (err) {
      if (err instanceof LettaRuleSolverError) {
        return {
          allowed: [],
          reason: err.message,
          requires_approval: [],
          uncalled_required: solver.getUncalledRequiredTools(effectiveHistory, availSet),
        };
      }
      throw err;
    }
  }

  async validateSequence(
    taskType: string,
    sequence: string[],
  ): Promise<SequenceValidationResult> {
    const solver = this.getSolver(taskType);
    if (!solver) {
      return { valid: true, violations: [] };
    }

    return solver.validateSequence(sequence, this.getAllToolNames());
  }

  /** Get compiled prompt summary of rules for a task type. */
  compilePrompt(taskType: string): string | null {
    const solver = this.getSolver(taskType);
    if (!solver) return null;
    return solver.compilePrompt();
  }

  private getAllToolNames(): Set<string> {
    const names = new Set<string>();
    for (const rules of this.rulesByTaskType.values()) {
      for (const rule of rules) {
        names.add(rule.tool_name);
        if ("children" in rule && Array.isArray(rule.children)) {
          for (const child of rule.children) names.add(child);
        }
        if ("child_output_mapping" in rule && rule.child_output_mapping) {
          for (const child of Object.values(rule.child_output_mapping)) names.add(child);
        }
        if ("default_child" in rule && rule.default_child) names.add(rule.default_child);
      }
    }
    return names;
  }

  /** Build an approximate linear sequence from child rules. Used for ToolRule compatibility. */
  private buildSequenceFromChildRules(rules: LettaToolRule[]): string[] {
    const childRules = rules.filter(
      (r): r is Extract<LettaToolRule, { type: "constrain_child_tools" }> =>
        r.type === "constrain_child_tools",
    );
    if (childRules.length === 0) return [];

    const adj = new Map<string, string>();
    for (const rule of childRules) {
      if (rule.children.length > 0) adj.set(rule.tool_name, rule.children[0]);
    }

    const initRules = rules.filter((r) => r.type === "run_first");
    const allChildren = new Set(childRules.flatMap((r) => r.children));
    let start: string | undefined;
    if (initRules.length > 0) {
      start = initRules[0].tool_name;
    } else {
      for (const parent of adj.keys()) {
        if (!allChildren.has(parent)) {
          start = parent;
          break;
        }
      }
    }
    if (!start) return [];

    const sequence: string[] = [start];
    const visited = new Set([start]);
    let current = start;
    while (adj.has(current)) {
      const next = adj.get(current)!;
      if (visited.has(next)) break;
      sequence.push(next);
      visited.add(next);
      current = next;
    }
    return sequence;
  }
}
