/**
 * Composer autocomplete. Port of app.js:12946-13033, 13125 + the keydown
 * branch at 13403-13411.
 *
 * `@` files (one artifact-index page + this session, version-pinned),
 * `#` sessions, `/` skills. Popup is `#composer-ac`. `ac` is the live
 * controller hung on window so F-11's send() keydown can see `ac.open`.
 *
 * `@` does not keep a project-wide cache and does not call
 * `GET /projects/{pid}/artifacts`. A page is one filename query, limit 20.
 * A response whose project, session, or update generation no longer matches
 * is discarded. Hidden rows (`priority < 0`) stay out.
 */

import { t } from "../../i18n/runtime";
import { artifacts } from "../../stores/artifacts";
import { currentId, sessions } from "../../stores/session";
import { fetchArtifactIndexPage } from "../artifacts/api";
import { filesT } from "../artifacts/copy";
import { effProject } from "../customize/host";
import { $, el, grow } from "../sessions/dom";
import { renderComposerRefChips } from "../sessions/transcript";
import { loadSkillsCatalog } from "./catalog";
import { acDetectFrom, type ComposerDetect } from "./detect";
import {
  artifactToAcItem,
  mergeArtifactCandidates,
  rankComposerItems,
  sessionToAcItem,
  skillToAcItem,
  type AcItem,
  type ArtifactLike,
} from "./rank";

export type AcState = {
  open: boolean;
  items: AcItem[];
  idx: number;
  trigger: string;
  start: number;
};

export const ac: AcState = {
  open: false,
  items: [],
  idx: 0,
  trigger: "",
  start: 0,
};

/** Input settles for this long before an `@` index request. `#` and `/` do not wait. */
export const AC_DEBOUNCE_MS = 150;
/** Rows asked of artifact-index. The popup still caps at `AC_LIMIT` (8). */
export const AC_INDEX_LIMIT = 20;

type AcNotice = "loading" | "error" | "recent";

const NOTICE_KEY: Record<AcNotice, string> = {
  loading: "ac.files.searching",
  error: "ac.files.failed",
  recent: "ac.files.recent",
};

type AcToken = {
  pid: string;
  sessionKey: string;
  query: string;
  seq: number;
};

type AcDecision =
  | { kind: "apply"; now: ComposerDetect }
  | { kind: "drop" }
  | { kind: "local" }
  | { kind: "close" };

/** Bumped by every `acUpdate`. A load that settles after a newer one started is dropped. */
let acSeq = 0;
let acNotice: AcNotice | null = null;
let acTimer: ReturnType<typeof setTimeout> | null = null;
let acAbort: AbortController | null = null;
let debounceResolve: (() => void) | null = null;

function projectId(): string {
  return effProject() || "";
}

function sessionKey(): string {
  return currentId.value || "";
}

function isShownArtifact(row: ArtifactLike | null | undefined): row is ArtifactLike {
  if (!row || !row.filename) return false;
  const priority = (row as { priority?: number | null }).priority;
  return !(typeof priority === "number" && priority < 0);
}

function sessionCandidates(): ArtifactLike[] {
  return ((artifacts.value || []) as ArtifactLike[]).filter(isShownArtifact);
}

function toItems(rows: ArtifactLike[]): AcItem[] {
  return rows.map((a) => artifactToAcItem(a, currentId.value, t("ac.fromOtherSession")));
}

function isAbortError(err: unknown): boolean {
  return !!err && typeof err === "object" && (err as { name?: string }).name === "AbortError";
}

function clearDebounce(): void {
  if (acTimer !== null) {
    clearTimeout(acTimer);
    acTimer = null;
  }
  if (debounceResolve) {
    const resolve = debounceResolve;
    debounceResolve = null;
    resolve();
  }
}

function abortInFlight(): void {
  const ctrl = acAbort;
  acAbort = null;
  if (ctrl) ctrl.abort();
}

function cancelFileSearch(): void {
  clearDebounce();
  abortInFlight();
}

/**
 * A newer keystroke owns the popup: leave it alone.
 * The project or session moved under this same update: do not render the
 * page that was fetched for the old one.
 * The caret left the token this update searched: close.
 */
function decide(token: AcToken): AcDecision {
  if (token.seq !== acSeq) return { kind: "drop" };
  if (token.pid !== projectId() || token.sessionKey !== sessionKey()) return { kind: "local" };
  const now = acDetect();
  if (!now || now.trigger !== "@" || now.query !== token.query) return { kind: "close" };
  return { kind: "apply", now };
}

function settle(token: AcToken, rows: ArtifactLike[], notice: AcNotice | null): void {
  const decision = decide(token);
  if (decision.kind === "drop") return;
  if (decision.kind === "apply") {
    showFilePopup(rows, decision.now, notice);
    return;
  }
  if (decision.kind === "local") {
    const now = acDetect();
    if (!now || now.trigger !== "@") acClose();
    else showFilePopup(sessionCandidates(), now, null);
    return;
  }
  acClose();
}

export function acDetect(): ComposerDetect | null {
  const c = $("#composer") as HTMLTextAreaElement | null;
  if (!c) return null;
  const pos = c.selectionStart;
  const before = (c.value || "").slice(0, pos);
  return acDetectFrom(before, pos);
}

/**
 * One index page for `query` (empty → most recent page, no `q`), merged with
 * this session's visible files. No shared cache: callers that outlive their
 * `{pid, sessionKey, query, seq}` token drop the array.
 */
export async function acProjectFiles(query = "", signal?: AbortSignal): Promise<ArtifactLike[]> {
  const pid = projectId();
  const sessionList = sessionCandidates();
  if (!pid) return sessionList;
  const page = await fetchArtifactIndexPage(pid, {
    q: query,
    limit: AC_INDEX_LIMIT,
    signal,
  });
  return mergeArtifactCandidates(page.artifacts.filter(isShownArtifact), sessionList);
}

export function ensureComposerAc(): HTMLElement | null {
  if (typeof document === "undefined") return null;
  let box = document.getElementById("composer-ac");
  if (box) return box;
  box = el("div", "composer-ac hidden");
  box.id = "composer-ac";
  const refs = document.getElementById("composer-refs");
  const hint = document.getElementById("composer-hint");
  const composer = document.getElementById("composer");
  const parent =
    (refs && refs.parentNode) ||
    (hint && hint.parentNode) ||
    (composer && composer.parentNode) ||
    document.body;
  if (refs && refs.parentNode === parent) parent.insertBefore(box, refs.nextSibling);
  else if (hint && hint.parentNode === parent) parent.insertBefore(box, hint);
  else parent.appendChild(box);
  return box;
}

export function acClose(): void {
  ac.open = false;
  ac.items = [];
  ac.idx = 0;
  acNotice = null;
  cancelFileSearch();
  const b = $("#composer-ac");
  if (!b) return;
  b.classList.add("hidden");
  b.innerHTML = "";
}

function showFilePopup(rows: ArtifactLike[], d: ComposerDetect, notice: AcNotice | null): void {
  const items = rankComposerItems(toItems(rows), d.query);
  if (!items.length && !notice) {
    acClose();
    return;
  }
  acNotice = notice;
  ac.items = items;
  ac.idx = 0;
  ac.trigger = d.trigger;
  ac.start = d.start;
  ac.open = items.length > 0;
  acRender();
}

export function acRender(): void {
  const box = ensureComposerAc();
  if (!box) return;
  box.innerHTML = "";
  ac.items.forEach((it, i) => {
    const row = el("div", "ac-item" + (i === ac.idx ? " on" : ""));
    row.appendChild(el("span", "ac-lbl", ac.trigger + (it.label || "")));
    if (it.sub) row.appendChild(el("span", "ac-sub", it.sub));
    row.onmousedown = (e) => {
      e.preventDefault();
      acPick(i);
    };
    box.appendChild(row);
  });
  if (acNotice) box.appendChild(el("div", "ac-hint", filesT(NOTICE_KEY[acNotice])));
  box.classList.remove("hidden");
}

export function acPick(i: number): void {
  const it = ac.items[i];
  if (!it) return;
  const c = $("#composer") as HTMLTextAreaElement | null;
  if (!c) return;
  const val = c.value;
  const pos = c.selectionStart;
  const token = ac.trigger + it.insert + " ";
  c.value = val.slice(0, ac.start) + token + val.slice(pos);
  const np = ac.start + token.length;
  c.setSelectionRange(np, np);
  acClose();
  grow();
  renderComposerRefChips();
  c.focus();
}

async function runFileSearch(seq: number): Promise<void> {
  if (seq !== acSeq) return;
  const d = acDetect();
  if (!d || d.trigger !== "@") {
    acClose();
    return;
  }
  const token: AcToken = {
    pid: projectId(),
    sessionKey: sessionKey(),
    query: d.query,
    seq,
  };
  if (!token.pid) {
    settle(token, sessionCandidates(), null);
    return;
  }
  showFilePopup(sessionCandidates(), d, "loading");
  const ctrl = new AbortController();
  acAbort = ctrl;
  try {
    const merged = await acProjectFiles(token.query, ctrl.signal);
    settle(token, merged, token.query ? null : "recent");
  } catch (err) {
    if (isAbortError(err)) return;
    settle(token, sessionCandidates(), "error");
  } finally {
    if (acAbort === ctrl) acAbort = null;
  }
}

function scheduleFileSearch(seq: number): Promise<void> {
  clearDebounce();
  abortInFlight();
  const d = acDetect();
  if (!d || d.trigger !== "@" || seq !== acSeq) return Promise.resolve();
  // Session rows are local. With no project there is no index to query.
  showFilePopup(sessionCandidates(), d, null);
  if (!projectId()) return Promise.resolve();
  return new Promise((resolve) => {
    debounceResolve = resolve;
    acTimer = setTimeout(() => {
      acTimer = null;
      debounceResolve = null;
      void runFileSearch(seq).then(resolve, resolve);
    }, AC_DEBOUNCE_MS);
  });
}

async function finishOther(seq: number, d: ComposerDetect): Promise<void> {
  let items: AcItem[] = [];
  if (d.trigger === "#") {
    const rows = (sessions.value || []) as Array<{
      name?: string;
      task_summary?: string;
    }>;
    items = rows.map(sessionToAcItem);
  } else if (d.trigger === "/") {
    // A failed read offers nothing this time; the next `/` asks again.
    const sk = await loadSkillsCatalog().catch(() => []);
    items = sk.map(skillToAcItem);
  }
  // The skill list loads asynchronously. Keystrokes in the meantime started
  // newer updates, and the caret may have moved without one.
  if (seq !== acSeq) return;
  const now = acDetect();
  if (!now || now.trigger !== d.trigger) {
    acClose();
    return;
  }
  items = rankComposerItems(items, now.query);
  acNotice = null;
  if (!items.length) {
    acClose();
    return;
  }
  ac.open = true;
  ac.items = items;
  ac.idx = 0;
  ac.trigger = now.trigger;
  ac.start = now.start;
  acRender();
}

export function acUpdate(): Promise<void> {
  const seq = ++acSeq;
  const d = acDetect();
  if (!d) {
    acClose();
    return Promise.resolve();
  }
  if (d.trigger !== "@") {
    cancelFileSearch();
    acNotice = null;
    return finishOther(seq, d);
  }
  return scheduleFileSearch(seq);
}

function onComposerKeydown(e: KeyboardEvent): void {
  if (e.isComposing || e.keyCode === 229) return;
  if (!ac.open && !acNotice) return;
  if (e.key === "Escape") {
    e.preventDefault();
    e.stopImmediatePropagation();
    acClose();
    return;
  }
  if (!ac.open || ac.items.length === 0) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    e.stopImmediatePropagation();
    ac.idx = (ac.idx + 1) % ac.items.length;
    acRender();
    return;
  }
  if (e.key === "ArrowUp") {
    e.preventDefault();
    e.stopImmediatePropagation();
    ac.idx = (ac.idx - 1 + ac.items.length) % ac.items.length;
    acRender();
    return;
  }
  if (e.key === "Enter" || e.key === "Tab") {
    e.preventDefault();
    e.stopImmediatePropagation();
    acPick(ac.idx);
    return;
  }
}

let composerBound = false;

function attachComposer(c: HTMLTextAreaElement): void {
  if (composerBound) return;
  composerBound = true;
  ensureComposerAc();
  c.addEventListener("input", () => {
    void acUpdate();
  });
  c.addEventListener("keydown", onComposerKeydown, true);
  c.addEventListener("blur", () => setTimeout(acClose, 120));
}

export function bindComposerAutocomplete(): void {
  if (typeof document === "undefined" || composerBound) return;
  const tryBind = (): void => {
    const c = document.getElementById("composer") as HTMLTextAreaElement | null;
    if (c) attachComposer(c);
  };
  tryBind();
  if (composerBound) return;
  if (typeof MutationObserver === "function") {
    const ob = new MutationObserver(() => {
      tryBind();
      if (composerBound) ob.disconnect();
    });
    ob.observe(document.documentElement || document.body, {
      childList: true,
      subtree: true,
    });
  }
}
