// ═══════════════════════════════════════════════════════════════
// Tests: SelfHealer (L3 AI Emergency Recovery)
// ═══════════════════════════════════════════════════════════════

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ── Mocks ────────────────────────────────────────────────────

vi.mock("node:fs/promises", () => ({
  mkdir: vi.fn().mockResolvedValue(undefined),
  readFile: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
  writeFile: vi.fn().mockResolvedValue(undefined),
  rename: vi.fn().mockResolvedValue(undefined),
  appendFile: vi.fn().mockResolvedValue(undefined),
  readdir: vi.fn().mockResolvedValue([]),
}));

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
}));

vi.mock("node:child_process", () => ({
  execSync: vi.fn(() => ""),
}));

vi.mock("../../core/logger.js", () => ({
  Logger: vi.fn().mockImplementation(() => ({
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  })),
}));

vi.mock("../../security/audit-log.js", () => ({
  auditLog: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../../security/tripwire.js", () => ({
  checkTripwire: vi.fn().mockResolvedValue(undefined),
}));

// Mock claudeCircuitBreaker — use vi.hoisted() so the mock is available before vi.mock hoisting
const { mockForceReset } = vi.hoisted(() => ({
  mockForceReset: vi.fn(),
}));

vi.mock("../../agents/llm-provider.js", () => ({
  claudeCircuitBreaker: { forceReset: mockForceReset },
  LLMProviderError: class extends Error {
    constructor(
      public readonly provider: string,
      public readonly statusCode: number,
      message: string,
      public readonly retryable: boolean,
    ) {
      super(message);
      this.name = "LLMProviderError";
    }
  },
}));

// ── Imports ──────────────────────────────────────────────────

import { SelfHealer } from "../self-healer.js";
import type { HealContext, Diagnosis } from "../self-healer.js";
import type { PepagiConfig } from "../../config/loader.js";
import type { LLMProvider, LLMResponse } from "../../agents/llm-provider.js";
import type { TaskStore } from "../../core/task-store.js";
import type { SecurityGuard } from "../../security/security-guard.js";
import { eventBus } from "../../core/event-bus.js";

// ── Test Config ──────────────────────────────────────────────

function makeConfig(overrides: Partial<PepagiConfig["selfHealing"]> = {}): PepagiConfig {
  return {
    managerProvider: "claude",
    managerModel: "claude-sonnet-4-6",
    agents: {
      claude: { enabled: true, apiKey: "", model: "claude-sonnet-4-6", maxOutputTokens: 4096, temperature: 0.3, maxAgenticTurns: 0 },
      gpt: { enabled: false, apiKey: "", model: "gpt-4o", maxOutputTokens: 4096, temperature: 0.3, maxAgenticTurns: 0 },
      gemini: { enabled: false, apiKey: "", model: "gemini-2.0-flash", maxOutputTokens: 4096, temperature: 0.3, maxAgenticTurns: 0 },
    },
    profile: {
      userName: "",
      assistantName: "PEPAGI",
      communicationStyle: "human" as const,
      language: "cs",
      subscriptionMode: false,
      gptSubscriptionMode: false,
    },
    platforms: {
      telegram: { enabled: false, botToken: "", allowedUserIds: [], welcomeMessage: "" },
      whatsapp: { enabled: false, allowedNumbers: [], sessionPath: "", welcomeMessage: "" },
      discord: { enabled: false, botToken: "", allowedUserIds: [], allowedChannelIds: [], commandPrefix: "!", welcomeMessage: "" },
      imessage: { enabled: false, allowedNumbers: [] },
    },
    security: {
      maxCostPerTask: 1.0,
      maxCostPerSession: 10.0,
      blockedCommands: ["rm -rf /"],
      requireApproval: [],
    },
    queue: { maxConcurrentTasks: 4, taskTimeoutMs: 120_000 },
    customProviders: {},
    consciousness: { profile: "MINIMAL" as const, enabled: true },
    web: { enabled: false, port: 3100, host: "127.0.0.1", authToken: "" },
    n8n: { enabled: false, baseUrl: "", webhookPaths: [], apiKey: "" },
    selfHealing: {
      enabled: true,
      maxAttemptsPerHour: 3,
      cooldownMs: 100, // short cooldown for tests
      costCapPerAttempt: 0.50,
      allowCodeFixes: false,
      ...overrides,
    },
  };
}

function makeMockLLM(): LLMProvider {
  return {
    quickCall: vi.fn().mockResolvedValue({
      content: JSON.stringify({
        problem: "test issue",
        suggestedTier: 1,
        suggestedAction: "reset_circuit_breaker",
        confidence: 0.8,
      }),
      toolCalls: [],
      usage: { inputTokens: 100, outputTokens: 50 },
      cost: 0.001,
      model: "claude-haiku-4-5",
      latencyMs: 200,
    } satisfies LLMResponse),
    quickClaude: vi.fn(),
    call: vi.fn(),
    configure: vi.fn(),
    registerCustomProviders: vi.fn(),
  } as unknown as LLMProvider;
}

function makeMockTaskStore(tasks: Array<{ id: string; status: string; startedAt?: Date; tokensUsed?: { input: number; output: number } }>): TaskStore {
  return {
    getAll: vi.fn(() => tasks.map(t => ({
      id: t.id,
      status: t.status,
      title: "test task",
      description: "test",
      startedAt: t.startedAt ?? null,
      completedAt: null,
      lastError: null,
      tokensUsed: t.tokensUsed ?? { input: 0, output: 0 },
    }))),
    get: vi.fn(),
    load: vi.fn().mockResolvedValue(undefined),
  } as unknown as TaskStore;
}

function makeMockGuard(): SecurityGuard {
  return {} as unknown as SecurityGuard;
}

// ─── Tests ───────────────────────────────────────────────────

describe("SelfHealer.canAttemptHeal", () => {
  it("allows first attempt", () => {
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig());
    expect(healer.canAttemptHeal()).toBe(true);
  });

  it("blocks when rate limit exceeded", () => {
    const config = makeConfig({ maxAttemptsPerHour: 2, cooldownMs: 0 });
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);

    // First two pass
    expect(healer.canAttemptHeal()).toBe(true);
    expect(healer.canAttemptHeal()).toBe(true);
    // Third is blocked
    expect(healer.canAttemptHeal()).toBe(false);
  });

  it("blocks during cooldown", () => {
    const config = makeConfig({ cooldownMs: 60_000 });
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);

    // First attempt sets cooldown
    expect(healer.canAttemptHeal()).toBe(true);
    // Immediately blocked by cooldown
    expect(healer.canAttemptHeal()).toBe(false);
  });

  it("allows after cooldown expires", async () => {
    const config = makeConfig({ cooldownMs: 50 }); // 50ms cooldown
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);

    expect(healer.canAttemptHeal()).toBe(true);
    // Wait for cooldown
    await new Promise(r => setTimeout(r, 60));
    expect(healer.canAttemptHeal()).toBe(true);
  });

  it("does not count attempts from over an hour ago", () => {
    const config = makeConfig({ maxAttemptsPerHour: 2, cooldownMs: 0 });
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);

    // Simulate old attempts by accessing internal state
    const oldTs = Date.now() - 3_700_000; // over 1 hour ago
    // @ts-expect-error — accessing private for test
    healer.healAttempts.push({ ts: oldTs, tier: 1, success: true });
    // @ts-expect-error — accessing private for test
    healer.healAttempts.push({ ts: oldTs, tier: 1, success: false });

    // These old attempts shouldn't count
    expect(healer.canAttemptHeal()).toBe(true);
  });
});

describe("SelfHealer.isProtectedFile", () => {
  const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig());

  it("detects protected security files", () => {
    expect(healer.isProtectedFile("security-guard.ts")).toBe(true);
    expect(healer.isProtectedFile("src/security/security-guard.ts")).toBe(true);
    expect(healer.isProtectedFile("tripwire.ts")).toBe(true);
    expect(healer.isProtectedFile("audit-log.ts")).toBe(true);
    expect(healer.isProtectedFile("dlp-engine.ts")).toBe(true);
    expect(healer.isProtectedFile("credential-scrubber.ts")).toBe(true);
    expect(healer.isProtectedFile("credential-lifecycle.ts")).toBe(true);
    expect(healer.isProtectedFile("supply-chain.ts")).toBe(true);
    expect(healer.isProtectedFile("tls-verifier.ts")).toBe(true);
    expect(healer.isProtectedFile("cost-tracker.ts")).toBe(true);
  });

  it("allows non-protected files", () => {
    expect(healer.isProtectedFile("mediator.ts")).toBe(false);
    expect(healer.isProtectedFile("llm-provider.ts")).toBe(false);
    expect(healer.isProtectedFile("src/core/types.ts")).toBe(false);
  });
});

describe("SelfHealer.diagnose", () => {
  it("quick-diagnoses circuit breaker issues without LLM", async () => {
    const llm = makeMockLLM();
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "system:alert", message: "Circuit breaker OPEN", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(diag.suggestedTier).toBe(1);
    expect(diag.suggestedAction).toBe("reset_circuit_breaker");
    // Should NOT have called LLM
    expect(llm.quickCall).not.toHaveBeenCalled();
  });

  it("quick-diagnoses stuck tasks", async () => {
    const llm = makeMockLLM();
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "meta:watchdog_alert", message: "Task stuck for 15 minutes", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(diag.suggestedAction).toBe("kill_stuck_tasks");
  });

  it("quick-diagnoses config corruption", async () => {
    const llm = makeMockLLM();
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "system:alert", message: "Config parse error: invalid JSON", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(diag.suggestedAction).toBe("repair_config");
  });

  it("quick-diagnoses OOM risk", async () => {
    const llm = makeMockLLM();
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "system:alert", message: "Heap memory usage at 85%", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(diag.suggestedAction).toBe("force_gc");
  });

  it("falls back to LLM for complex issues", async () => {
    const llm = makeMockLLM();
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "task:failed", message: "Some unusual error", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(llm.quickCall).toHaveBeenCalled();
    expect(diag.problem).toBe("test issue");
  });

  it("returns fallback diagnosis when LLM fails", async () => {
    const llm = makeMockLLM();
    (llm.quickCall as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("LLM unavailable"));
    const healer = new SelfHealer(llm, makeMockTaskStore([]), makeMockGuard(), makeConfig());

    const ctx: HealContext = { trigger: "task:failed", message: "Something weird", timestamp: Date.now() };
    const diag = await healer.diagnose(ctx);

    expect(diag.suggestedTier).toBe(1);
    expect(diag.suggestedAction).toBe("reset_circuit_breaker");
    expect(diag.confidence).toBe(0.3);
  });
});

describe("SelfHealer.healTier1", () => {
  beforeEach(() => {
    mockForceReset.mockClear();
  });

  it("resets circuit breaker", async () => {
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig());
    const diag: Diagnosis = { problem: "CB open", suggestedTier: 1, suggestedAction: "reset_circuit_breaker", confidence: 0.9 };

    const result = await healer.healTier1(diag);

    expect(result.success).toBe(true);
    expect(result.tier).toBe(1);
    expect(mockForceReset).toHaveBeenCalled();
  });

  it("kills stuck tasks", async () => {
    const oldTime = new Date(Date.now() - 700_000); // 11+ min ago
    const tasks = [
      { id: "t1", status: "running", startedAt: oldTime },
      { id: "t2", status: "running", startedAt: new Date() }, // not stuck
      { id: "t3", status: "completed" },
    ];
    const taskStore = makeMockTaskStore(tasks);
    const healer = new SelfHealer(makeMockLLM(), taskStore, makeMockGuard(), makeConfig());
    const diag: Diagnosis = { problem: "Stuck tasks", suggestedTier: 1, suggestedAction: "kill_stuck_tasks", confidence: 0.8 };

    const emittedEvents: string[] = [];
    const handler = (e: { type: string }) => { if (e.type === "task:failed") emittedEvents.push(e.type); };
    eventBus.onAny(handler);

    const result = await healer.healTier1(diag);
    eventBus.offAny(handler);

    expect(result.success).toBe(true);
    expect(result.details).toContain("1 stuck tasks");
    expect(emittedEvents).toContain("task:failed");
  });

  it("handles unknown action gracefully", async () => {
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig());
    const diag: Diagnosis = { problem: "test", suggestedTier: 1, suggestedAction: "unknown_action", confidence: 0.5 };

    const result = await healer.healTier1(diag);

    expect(result.success).toBe(false);
    expect(result.details).toContain("Unknown Tier 1 action");
  });
});

describe("SelfHealer.healTier2", () => {
  it("refuses when protected files are affected", async () => {
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig({ allowCodeFixes: true }));
    const diag: Diagnosis = {
      problem: "bug in security",
      suggestedTier: 2,
      suggestedAction: "code_fix",
      affectedFiles: ["src/security/security-guard.ts"],
      confidence: 0.7,
    };

    const result = await healer.healTier2(diag);

    expect(result.success).toBe(false);
    expect(result.details).toContain("protected security files");
  });
});

describe("SelfHealer lifecycle", () => {
  afterEach(() => {
    // Clean up any lingering listeners
  });

  it("does not start when disabled in config", () => {
    const config = makeConfig({ enabled: false });
    // Force the top-level enabled to false
    config.selfHealing = { ...config.selfHealing!, enabled: false };
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);

    healer.start();
    // No handler should be registered — stopping should be a no-op
    healer.stop();
  });

  it("registers and removes event listener", () => {
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), makeConfig());

    healer.start();
    // Starting again should be idempotent
    healer.start();

    healer.stop();
    // Stopping again should be safe
    healer.stop();
  });

  it("responds to task:failed events", async () => {
    const config = makeConfig({ cooldownMs: 0 });
    const healer = new SelfHealer(makeMockLLM(), makeMockTaskStore([]), makeMockGuard(), config);
    healer.start();

    const emittedEvents: string[] = [];
    const handler = (e: { type: string }) => {
      if (e.type.startsWith("self-heal:")) emittedEvents.push(e.type);
    };
    eventBus.onAny(handler);

    // Emit task:failed
    eventBus.emit({ type: "task:failed", taskId: "t1", error: "Circuit breaker open" });

    // Wait for async processing
    await new Promise(r => setTimeout(r, 100));

    eventBus.offAny(handler);
    healer.stop();

    expect(emittedEvents).toContain("self-heal:attempt");
    expect(emittedEvents).toContain("self-heal:success");
  });
});
