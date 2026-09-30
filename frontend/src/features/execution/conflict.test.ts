import { effect } from "@preact/signals";
import { afterEach, describe, expect, it, vi } from "vitest";
import { workbenchErrors } from "../../stores/timeline";
import { historyT } from "../messages/copy";
import { scheduleWorkbenchRefresh } from "../notebook/kernel";
import { setExecutionFetch } from "./api";
import { applyForkPresentation, forkFromMessage } from "./branch";
import {
  FORK_NO_CHECKPOINT_MESSAGE,
  forkErrorDisplay,
  forkOnce,
  httpStatusOf,
  isForkNoCheckpoint,
  presentForkError,
  shouldRetryFork,
} from "./conflict";

vi.mock("../notebook/kernel", () => ({
  scheduleWorkbenchRefresh: vi.fn(),
}));

function conflict409(message = FORK_NO_CHECKPOINT_MESSAGE): {
  status: number;
  code: string;
  message: string;
  error: string;
} {
  return { status: 409, code: "conflict", message, error: message };
}

describe("fork 409 presentation (CursorCheckpointUnavailable)", () => {
  it("presents the server sentence for fork-without-checkpoint and never retries", () => {
    const presented = presentForkError(conflict409());
    expect(presented.kind).toBe("conflict");
    expect(presented.noCheckpoint).toBe(true);
    expect(presented.httpStatus).toBe(409);
    expect(presented.code).toBe("conflict");
    expect(presented.retry).toBe(false);
    expect(presented.masked).toBe(false);
    expect(shouldRetryFork(presented)).toBe(false);
    expect(forkErrorDisplay(presented)).toBe(FORK_NO_CHECKPOINT_MESSAGE);
    expect(forkErrorDisplay(presented)).not.toMatch(/try again/i);
    expect(forkErrorDisplay(presented)).not.toBe("");
  });

  it("recognises the 409 from notebookFetch, which only keeps the message", () => {
    const presented = presentForkError(new Error(FORK_NO_CHECKPOINT_MESSAGE));
    expect(presented.kind).toBe("conflict");
    expect(presented.noCheckpoint).toBe(true);
    expect(presented.retry).toBe(false);
    expect(forkErrorDisplay(presented)).toContain("no exact cursor checkpoint");
    expect(isForkNoCheckpoint(new Error(FORK_NO_CHECKPOINT_MESSAGE))).toBe(true);
  });

  it("does not treat a recovery domain status string as HTTP 409", () => {
    expect(httpStatusOf({ status: "failed", message: "partial restore" })).toBeNull();
    const presented = presentForkError({ status: "failed", message: "partial restore" });
    expect(presented.kind).toBe("error");
    expect(presented.noCheckpoint).toBe(false);
    expect(presented.retry).toBe(false);
  });

  it("other HTTP 409s are still conflicts: no retry, message not rewritten", () => {
    const presented = presentForkError({
      status: 409,
      code: "conflict",
      message: "session deletion is already in progress",
      error: "session deletion is already in progress",
    });
    expect(presented.kind).toBe("conflict");
    expect(presented.noCheckpoint).toBe(false);
    expect(presented.retry).toBe(false);
    expect(forkErrorDisplay(presented)).toBe("session deletion is already in progress");
  });

  it("forkOnce invokes the POST exactly once on 409", async () => {
    const post = vi.fn(async () => {
      throw conflict409();
    });
    const attempt = await forkOnce(post);
    expect(post).toHaveBeenCalledTimes(1);
    expect(attempt.ok).toBe(false);
    if (attempt.ok) throw new Error("expected failure");
    expect(attempt.attempts).toBe(1);
    expect(attempt.presentation.noCheckpoint).toBe(true);
    expect(attempt.presentation.retry).toBe(false);
    expect(shouldRetryFork(attempt.presentation)).toBe(false);
  });

  it("forkOnce does not retry a non-409 either", async () => {
    const post = vi.fn(async () => {
      throw { status: 404, message: "session not found", error: "session not found" };
    });
    const attempt = await forkOnce(post);
    expect(post).toHaveBeenCalledTimes(1);
    expect(attempt.ok).toBe(false);
    if (attempt.ok) throw new Error("expected failure");
    expect(attempt.presentation.kind).toBe("error");
    expect(attempt.presentation.retry).toBe(false);
  });

  it("does not map a 409 into a successful fork", async () => {
    const attempt = await forkOnce(async () => {
      throw conflict409();
    });
    expect(attempt.ok).toBe(false);
    expect("result" in attempt).toBe(false);
  });

  it("keeps a successful POST as ok without inventing a conflict", async () => {
    const attempt = await forkOnce(async () => ({ branch_id: "b1" }));
    expect(attempt).toEqual({ ok: true, result: { branch_id: "b1" }, attempts: 1 });
  });
});

describe("forkFromMessage", () => {
  const hints: string[] = [];

  afterEach(() => {
    hints.length = 0;
    vi.unstubAllGlobals();
    setExecutionFetch(null);
    vi.mocked(scheduleWorkbenchRefresh).mockClear();
  });

  function jsonResponse(body: unknown, status: number): Response {
    return new Response(JSON.stringify(body), { status });
  }

  it("posts exactly {from_message_id} once, then refreshes and names the branch", async () => {
    const posts: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal("hint", (message: string) => {
      hints.push(message);
    });
    setExecutionFetch(async (url, init) => {
      posts.push({ url: String(url), body: JSON.parse(String(init?.body || "{}")) });
      return jsonResponse({ branch_id: "br-1", name: "from question" }, 200);
    });
    const result = await forkFromMessage("frame/1", "msg-9");
    expect(posts).toEqual([{
      url: "/api/v1/frames/frame%2F1/branches/fork",
      body: { from_message_id: "msg-9" },
    }]);
    expect(result).toEqual({ ok: true, branch_id: "br-1", name: "from question" });
    expect(hints).toEqual([historyT("history.forkMessage.created", "from question")]);
    expect(scheduleWorkbenchRefresh).toHaveBeenCalledTimes(1);
  });

  it("presents a 409 once and does not fork again or fall back to another source", async () => {
    const posts: unknown[] = [];
    setExecutionFetch(async (_url, init) => {
      posts.push(JSON.parse(String(init?.body || "{}")));
      return jsonResponse({ error: FORK_NO_CHECKPOINT_MESSAGE, code: "conflict" }, 409);
    });
    const result = await forkFromMessage("frame-1", "msg-9");
    expect(posts).toEqual([{ from_message_id: "msg-9" }]);
    expect(result).toMatchObject({
      ok: false,
      presentation: {
        kind: "conflict",
        noCheckpoint: true,
        httpStatus: 409,
        retry: false,
        message: FORK_NO_CHECKPOINT_MESSAGE,
      },
    });
    expect(workbenchErrors.value.branchAction).toBe(FORK_NO_CHECKPOINT_MESSAGE);
    expect(scheduleWorkbenchRefresh).not.toHaveBeenCalled();
  });

  it("uses the history fallback when the failure has no sentence", async () => {
    setExecutionFetch(async () => {
      throw new Error("");
    });
    const result = await forkFromMessage("frame-1", "msg-9");
    expect(result).toMatchObject({
      ok: false,
      presentation: { message: historyT("history.forkMessage.failed"), retry: false },
    });
    expect(workbenchErrors.value.branchAction).toBe(historyT("history.forkMessage.failed"));
  });

  it("a second call while the first is in flight does not post", async () => {
    const posts: unknown[] = [];
    let release: (response: Response) => void = () => undefined;
    setExecutionFetch((_url, init) => {
      posts.push(JSON.parse(String(init?.body || "{}")));
      return new Promise<Response>((resolve) => {
        release = resolve;
      });
    });
    const first = forkFromMessage("frame-1", "msg-9");
    const second = forkFromMessage("frame-1", "msg-9");
    expect(posts).toEqual([{ from_message_id: "msg-9" }]);
    expect(await second).toBeNull();
    release(jsonResponse({ branch_id: "br-2" }, 200));
    expect(await first).toMatchObject({ ok: true, branch_id: "br-2", name: "br-2" });
    expect(posts).toHaveLength(1);
    expect(scheduleWorkbenchRefresh).toHaveBeenCalledTimes(1);
  });
});

describe("applyForkPresentation", () => {
  it("publishes a new workbenchErrors object, so a subscriber sees the banner", () => {
    workbenchErrors.value = { recoveryAction: "earlier" };
    const seen: Array<Record<string, unknown>> = [];
    const stop = effect(() => {
      seen.push(workbenchErrors.value);
    });
    applyForkPresentation(presentForkError(conflict409()));
    stop();
    expect(seen).toHaveLength(2);
    expect(seen[1]).toEqual({ recoveryAction: "earlier", branchAction: FORK_NO_CHECKPOINT_MESSAGE });
  });
});
