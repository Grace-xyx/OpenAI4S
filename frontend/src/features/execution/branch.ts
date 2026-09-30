/**
 * Recovery / branch control REST. F-15 already paints the Timeline panels
 * (sanitize* 3032-3149 + island buttons). This lane owns the 409 presentation
 * for fork-without-checkpoint: one POST, surface the server sentence, never
 * retry, never rewrite as success.
 *
 * Routes (unchanged):
 *   POST /frames/{id}/branches/fork
 *   POST /frames/{id}/branches/checkpoints
 *   POST /frames/{id}/recovery/actions/{restore|retry|restart_fresh}
 */

import { workbenchErrors } from "../../stores/timeline";
import { isReady } from "../../compat/stub";
import { t } from "../../i18n/runtime";
import { historyT } from "../messages/copy";
import { publicText } from "../scrub/scrub";
import { api } from "./api";
import {
  forkErrorDisplay,
  forkOnce,
  presentForkError,
  type ForkPresentation,
} from "./conflict";

function hint(msg: string, err?: boolean): void {
  const fn = (globalThis as unknown as { hint?: unknown }).hint;
  if (isReady(fn)) (fn as (m: string, e?: boolean) => void)(msg, err);
}

/** Write the honest server sentence onto the Timeline branch-error banner. */
export function applyForkPresentation(presentation: ForkPresentation): void {
  // A new object, so a subscriber to workbenchErrors sees the banner change.
  workbenchErrors.value = {
    ...(workbenchErrors.value || {}),
    branchAction: forkErrorDisplay(presentation),
  };
  hint(t("branch.actionFailed", forkErrorDisplay(presentation)), true);
}

export async function forkFromCell(frameId: string, cellId: string): Promise<ForkPresentation | null> {
  const attempt = await forkOnce(() =>
    api(`/frames/${encodeURIComponent(frameId)}/branches/fork`, {
      method: "POST",
      body: JSON.stringify({ from_cell_id: cellId }),
    }),
  );
  if (attempt.ok) return null;
  applyForkPresentation(attempt.presentation);
  return attempt.presentation;
}

export type ForkResult =
  | { ok: true; branch_id: string; name: string }
  | { ok: false; presentation: ForkPresentation };

/** One in-flight fork per message. A second click must not POST again. */
const messageForkInFlight = new Set<string>();

function messageForkKey(frameId: string, messageId: string): string {
  return frameId + "\0" + messageId;
}

function branchLabel(result: unknown): { branch_id: string; name: string } {
  const rec = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
  const branchId = publicText(rec.branch_id, 96);
  const named = publicText(rec.name, 120);
  return { branch_id: branchId, name: named || branchId };
}

/**
 * Fork from one stored user message. One POST, body exactly
 * `{from_message_id}`. A 409 is the server's sentence, not a retry and not
 * a fork of the latest state. The new branch stays inactive.
 */
export async function forkFromMessage(
  frameId: string,
  messageId: string,
): Promise<ForkResult | null> {
  if (!frameId || !messageId) return null;
  const key = messageForkKey(frameId, messageId);
  if (messageForkInFlight.has(key)) return null;
  messageForkInFlight.add(key);
  try {
    const attempt = await forkOnce(() =>
      api(`/frames/${encodeURIComponent(frameId)}/branches/fork`, {
        method: "POST",
        body: JSON.stringify({ from_message_id: messageId }),
      }),
    );
    if (!attempt.ok) {
      const presentation = attempt.presentation.message
        ? attempt.presentation
        : { ...attempt.presentation, message: historyT("history.forkMessage.failed") };
      applyForkPresentation(presentation);
      return { ok: false, presentation };
    }
    const created = branchLabel(attempt.result);
    hint(historyT("history.forkMessage.created", created.name));
    try {
      const { scheduleWorkbenchRefresh } = await import("../notebook/kernel");
      scheduleWorkbenchRefresh();
    } catch {
      // The branch already exists. A failed panel refresh must not look like
      // a failed fork, and must not be retried as a second POST.
    }
    return { ok: true, ...created };
  } finally {
    messageForkInFlight.delete(key);
  }
}

export async function forkFromCheckpoint(
  frameId: string,
  checkpointId: string,
  name?: string,
): Promise<ForkPresentation | null> {
  const body: Record<string, string> = { from_checkpoint_id: checkpointId };
  if (name) body.name = name;
  const attempt = await forkOnce(() =>
    api(`/frames/${encodeURIComponent(frameId)}/branches/fork`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  );
  if (attempt.ok) return null;
  applyForkPresentation(attempt.presentation);
  return attempt.presentation;
}

export async function postRecoveryAction(
  frameId: string,
  actionId: string,
  branchId: string,
  confirm: boolean,
): Promise<{ ok: true } | { ok: false; message: string }> {
  try {
    await api(`/frames/${encodeURIComponent(frameId)}/recovery/actions/${encodeURIComponent(actionId)}`, {
      method: "POST",
      body: JSON.stringify({ branch_id: branchId, confirm }),
    });
    return { ok: true };
  } catch (error) {
    const presented = presentForkError(error);
    return { ok: false, message: publicText(presented.message, 240) };
  }
}
