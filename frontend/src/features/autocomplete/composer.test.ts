import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { artifacts } from "../../stores/artifacts";
import { skillsCatalog } from "../../stores/customize";
import { currentId, project, sessions } from "../../stores/session";
import { resetStoreFields } from "../../stores/signal-field";
import { filesT } from "../artifacts/copy";
import { loadSkillsCatalog } from "./catalog";
import { AC_DEBOUNCE_MS, AC_INDEX_LIMIT, ac, acClose, acPick, acUpdate } from "./composer";

/** Just enough DOM for the composer popup (no jsdom here). */
class El {
  id = "";
  className = "";
  textContent = "";
  children: El[] = [];
  parentNode: El | null = null;
  style: Record<string, string> = {};
  onmousedown: unknown = null;
  classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
  set innerHTML(_v: string) {
    this.children = [];
  }
  appendChild(child: El): El {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  insertBefore(child: El): El {
    return this.appendChild(child);
  }
}

class Composer extends El {
  value = "";
  selectionStart = 0;
  scrollHeight = 40;
  setSelectionRange(start: number): void {
    this.selectionStart = start;
  }
  focus(): void {}
}

let composer: Composer;
let popup: El | null;

function type(value: string, caret = value.length): void {
  composer.value = value;
  composer.selectionStart = caret;
}

type Pending = {
  url: string;
  resolve: (body: unknown) => void;
  reject: (err: unknown) => void;
};

let inflight: Pending[] = [];

/** Each artifact-index request waits until the test answers it. Abort is ignored. */
function pendingIndex(): Pending[] {
  inflight = [];
  vi.stubGlobal("fetch", (input: unknown) => {
    return new Promise((resolve, reject) => {
      let settled = false;
      const pending: Pending = {
        url: String(input),
        resolve: (body) => {
          if (settled) return;
          settled = true;
          resolve({
            ok: true,
            status: 200,
            text: () => Promise.resolve(JSON.stringify(body)),
          });
        },
        reject: (err) => {
          if (settled) return;
          settled = true;
          reject(err);
        },
      };
      inflight.push(pending);
    });
  });
  return inflight;
}

function page(rows: unknown[]): unknown {
  return { artifacts: rows, next_cursor: null, has_more: false };
}

function art(partial: { id: string; filename: string; version_id?: string; priority?: number }): Record<string, unknown> {
  return { priority: 0, version_id: "", artifact_id: partial.id, ...partial };
}

const FILES = [
  art({ id: "plot", filename: "plot.png" }),
  art({ id: "pca", filename: "pca.csv" }),
  art({ id: "plan", filename: "plan.md" }),
];
const SKILLS = [{ name: "plot" }, { name: "pca" }, { name: "plan" }];

function hintText(): string {
  const node = (popup?.children || []).find((child) => String(child.className).split(/\s+/).includes("ac-hint"));
  return node ? node.textContent : "";
}

function labels(): string[] {
  return ac.items.map((item) => item.label);
}

beforeEach(() => {
  vi.useFakeTimers();
  resetStoreFields();
  acClose();
  inflight = [];
  composer = new Composer();
  composer.id = "composer";
  const parent = new El();
  parent.appendChild(composer);
  popup = null;
  vi.stubGlobal("document", {
    querySelector: (sel: string) => (sel === "#composer" ? composer : sel === "#composer-ac" ? popup : null),
    getElementById: (id: string) => (id === "composer" ? composer : id === "composer-ac" ? popup : null),
    createElement: () => {
      const node = new El();
      if (!popup) popup = node;
      return node;
    },
    body: new El(),
  });
});

afterEach(() => {
  for (const pending of inflight) {
    pending.reject(Object.assign(new Error("cancelled"), { name: "AbortError" }));
  }
  inflight = [];
  acClose();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("composer @ mentions use one artifact-index page", () => {
  it("requests the encoded project index for the filename and never the full array", async () => {
    project.value = "a/b c";
    currentId.value = "sess";
    const pending = pendingIndex();
    type("@abc");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS - 1);
    expect(pending).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(pending).toHaveLength(1);
    expect(AC_INDEX_LIMIT).toBe(20);
    const url = pending[0]!.url;
    expect(url).toContain("/projects/a%2Fb%20c/artifact-index?q=abc&limit=20");
    expect(url).not.toMatch(/\/artifacts(?:\?|$)/);
    pending[0]!.resolve(page([]));
    await update;
  });

  it("shows at most 8 of a 20-row page", async () => {
    project.value = "proj";
    const rows = Array.from({ length: 20 }, (_, i) =>
      art({ id: `id-${i}`, filename: `hit-${String(i).padStart(2, "0")}.txt`, version_id: `v-${i}` }),
    );
    const pending = pendingIndex();
    type("@hit");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(1);
    expect(pending[0]!.url).toContain("limit=20");
    pending[0]!.resolve(page(rows));
    await update;
    expect(labels()).toEqual(rows.slice(0, 8).map((row) => String(row.filename)));
    expect(ac.items).toHaveLength(8);
  });

  it("drops a late page for an older query", async () => {
    project.value = "proj-order";
    currentId.value = "sess";
    const pending = pendingIndex();
    type("@ab");
    const older = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(1);
    type("@abc");
    const newer = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(2);
    pending[1]!.resolve(page([art({ id: "new", filename: "abc-new.txt", version_id: "v-new" })]));
    await newer;
    pending[0]!.resolve(page([art({ id: "old", filename: "ab-old.txt", version_id: "v-old" })]));
    await older;
    expect(ac.open).toBe(true);
    expect(labels()).toEqual(["abc-new.txt"]);
    expect(ac.items.map((item) => item.insert)).toEqual(["abc-new.txt#v-new"]);
  });

  it("does not show a page fetched for a project the user has left", async () => {
    project.value = "proj/1";
    currentId.value = "sess";
    artifacts.value = [art({ id: "local", filename: "ab-local.txt", version_id: "v-local" })];
    const pending = pendingIndex();
    type("@ab");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(1);
    expect(pending[0]!.url).toContain("/projects/proj%2F1/artifact-index?");
    expect(hintText()).toBe(filesT("ac.files.searching"));
    expect(labels()).toEqual(["ab-local.txt"]);
    project.value = "proj/2";
    pending[0]!.resolve(page([art({ id: "old", filename: "ab-old.txt", version_id: "v-old" })]));
    await update;
    expect(labels()).toEqual(["ab-local.txt"]);
    expect(labels().join(" ")).not.toContain("ab-old");
    expect(hintText()).toBe("");
    expect(pending).toHaveLength(1);
  });

  it("does not show a page fetched for a session the user has left", async () => {
    project.value = "proj";
    currentId.value = "sess-a";
    artifacts.value = [art({ id: "local", filename: "ab-local.txt", version_id: "v-local" })];
    const pending = pendingIndex();
    type("@ab");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    currentId.value = "sess-b";
    artifacts.value = [art({ id: "other", filename: "ab-other.txt", version_id: "v-other" })];
    pending[0]!.resolve(page([art({ id: "old", filename: "ab-server.txt", version_id: "v-server" })]));
    await update;
    expect(labels()).toEqual(["ab-other.txt"]);
    expect(hintText()).toBe("");
  });

  it("keeps this session's files and shows the error when the index request fails", async () => {
    project.value = "proj";
    currentId.value = "sess";
    artifacts.value = [art({ id: "local", filename: "abc-local.txt", version_id: "v-local" })];
    const urls: string[] = [];
    vi.stubGlobal("fetch", (input: unknown) => {
      urls.push(String(input));
      return Promise.reject(new Error("offline"));
    });
    type("@abc");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    await expect(update).resolves.toBeUndefined();
    expect(urls).toHaveLength(1);
    expect(urls[0]).toContain("/projects/proj/artifact-index?q=abc&limit=20");
    expect(urls[0]).not.toMatch(/\/artifacts(?:\?|$)/);
    expect(labels()).toEqual(["abc-local.txt"]);
    expect(hintText()).toBe(filesT("ac.files.failed"));
    expect(ac.open).toBe(true);
  });

  it("lists a shared artifact once, pinning the index page's version", async () => {
    project.value = "proj";
    artifacts.value = [art({ id: "a1", filename: "plot.png", version_id: "v-session" })];
    const pending = pendingIndex();
    type("@plot");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    pending[0]!.resolve(page([art({ id: "a1", filename: "plot.png", version_id: "v-server" })]));
    await update;
    expect(ac.items).toHaveLength(1);
    expect(ac.items[0]?.insert).toBe("plot.png#v-server");
  });

  it("inserts the exact version id", async () => {
    project.value = "proj";
    const pending = pendingIndex();
    type("@pl");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    pending[0]!.resolve(page([art({ id: "plot", filename: "plot.png", version_id: "v-f85486107c9c" })]));
    await update;
    expect(ac.items[0]?.insert).toBe("plot.png#v-f85486107c9c");
    acPick(0);
    expect(composer.value).toBe("@plot.png#v-f85486107c9c ");
  });

  it("omits q for an empty query and says the rows are only the recent page", async () => {
    project.value = "proj";
    const pending = pendingIndex();
    type("@");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(1);
    const params = new URL(pending[0]!.url, "https://fixture.invalid").searchParams;
    expect(params.has("q")).toBe(false);
    expect(params.get("limit")).toBe("20");
    pending[0]!.resolve(page([art({ id: "recent", filename: "recent.txt", version_id: "v-recent" })]));
    await update;
    expect(labels()).toEqual(["recent.txt"]);
    expect(hintText()).toBe(filesT("ac.files.recent"));
  });

  it("sends only the last request of a 5-character burst", async () => {
    project.value = "proj";
    const pending = pendingIndex();
    const updates: Promise<void>[] = [];
    for (const value of ["@a", "@ab", "@abc", "@abcd", "@abcde"]) {
      type(value);
      updates.push(acUpdate());
    }
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS - 1);
    expect(pending).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(pending).toHaveLength(1);
    expect(pending[0]!.url).toContain("q=abcde&limit=20");
    pending[0]!.resolve(page([]));
    await Promise.all(updates);
  });

  it("leaves hidden artifacts out of the popup", async () => {
    project.value = "proj";
    artifacts.value = [
      art({ id: "hidden-session", filename: "abc-hidden.txt", version_id: "v-h", priority: -1 }),
      art({ id: "shown-session", filename: "abc-shown.txt", version_id: "v-s", priority: 1 }),
    ];
    const pending = pendingIndex();
    type("@abc");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    pending[0]!.resolve(
      page([
        art({ id: "hidden-index", filename: "abc-secret.txt", version_id: "v-x", priority: -1 }),
        art({ id: "shown-index", filename: "abc-indexed.txt", version_id: "v-i", priority: 0 }),
      ]),
    );
    await update;
    expect(labels()).toEqual(["abc-indexed.txt", "abc-shown.txt"]);
  });

  it("shows the searching line with this session's files while the page is in flight", async () => {
    project.value = "proj";
    artifacts.value = [art({ id: "local", filename: "abc-local.txt", version_id: "v-local" })];
    const pending = pendingIndex();
    type("@abc");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(hintText()).toBe(filesT("ac.files.searching"));
    expect(labels()).toEqual(["abc-local.txt"]);
    pending[0]!.resolve(page([art({ id: "remote", filename: "abc-remote.txt", version_id: "v-remote" })]));
    await update;
    expect(labels()).toEqual(["abc-remote.txt", "abc-local.txt"]);
    expect(hintText()).toBe("");
  });
});

describe("composer autocomplete after an async load", () => {
  it("anchors on the token at the caret now, not the one read before the load", async () => {
    project.value = "proj-caret";
    const pending = pendingIndex();
    type("see @pl");
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(pending).toHaveLength(1);
    // Typed while the list was loading: text before the token shifts it.
    type("please see @pl");
    pending[0]!.resolve(page(FILES));
    await update;
    expect(ac.open).toBe(true);
    expect(ac.start).toBe("please see ".length);
    acPick(0);
    expect(composer.value).toBe("please see @plot.png ");
  });

  it("`/` completions come back after a failed catalog read", async () => {
    let up = false;
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        up
          ? { ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ skills: SKILLS })) }
          : { ok: false, status: 503, text: () => Promise.resolve('{"error":"catalog unavailable"}') },
      ),
    );
    type("/pl");
    await acUpdate();
    expect(ac.open).toBe(false);
    up = true;
    type("/pl");
    await acUpdate();
    expect(ac.open).toBe(true);
    expect(ac.items.map((it) => it.insert)).toEqual(["plot", "plan"]);
  });

  it("closes when the caret has left the token", async () => {
    project.value = "proj-left";
    const pending = pendingIndex();
    type("@pl and more", 3);
    const update = acUpdate();
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    composer.selectionStart = composer.value.length;
    pending[0]!.resolve(page(FILES));
    await update;
    expect(ac.open).toBe(false);
  });

  it("does not debounce # completions or ask the artifact index", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", (input: unknown) => {
      urls.push(String(input));
      return Promise.resolve({
        ok: true,
        status: 200,
        text: () => Promise.resolve(JSON.stringify({ skills: SKILLS })),
      });
    });
    sessions.value = [{ name: "alpha" }, { name: "alpine" }];
    type("#al");
    await acUpdate();
    expect(ac.open).toBe(true);
    expect(ac.items.map((item) => item.insert)).toEqual(["alpha", "alpine"]);
    expect(urls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(AC_DEBOUNCE_MS);
    expect(urls).toHaveLength(0);
  });
});

describe("filename-search hints", () => {
  it("keeps the Chinese and English strings, and does not call the search full-text", () => {
    const src = readFileSync(
      join(dirname(fileURLToPath(import.meta.url)), "../artifacts/copy.ts"),
      "utf8",
    );
    expect(src).toContain("正在搜索项目文件…");
    expect(src).toContain("项目文件搜索失败，仅显示本会话文件");
    expect(src).toContain("显示最近的项目文件，输入文件名可搜索全部");
    expect(src).toContain("Searching project files…");
    expect(src).toContain("Project file search failed. Showing only this session's files.");
    expect(src).toContain(
      "Showing the most recent project files. Type a filename to search all of them.",
    );
    expect(src.toLowerCase()).not.toContain("full-text");
    expect(src.toLowerCase()).not.toContain("full text");
    expect(filesT("ac.files.searching").length).toBeGreaterThan(0);
  });
});

describe("the shared skills catalog", () => {
  it("stores no failed read: the next caller asks again", async () => {
    let up = false;
    let requests = 0;
    vi.stubGlobal("fetch", () => {
      requests += 1;
      return Promise.resolve(
        up
          ? { ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ skills: SKILLS })) }
          : { ok: false, status: 503, text: () => Promise.resolve('{"error":"catalog unavailable"}') },
      );
    });
    await expect(loadSkillsCatalog()).rejects.toThrow(/catalog unavailable/);
    expect(skillsCatalog.value).toBeNull();
    up = true;
    await expect(loadSkillsCatalog()).resolves.toEqual(SKILLS);
    expect(requests).toBe(2);
    // A stored catalog, even an empty one, answers without a request.
    await loadSkillsCatalog();
    skillsCatalog.value = [];
    await expect(loadSkillsCatalog()).resolves.toEqual([]);
    expect(requests).toBe(2);
  });

  it("concurrent callers share the one request in flight", async () => {
    let answer: () => void = () => {};
    let requests = 0;
    vi.stubGlobal("fetch", () => {
      requests += 1;
      return new Promise((resolve) => {
        answer = () =>
          resolve({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ skills: SKILLS })) });
      });
    });
    const first = loadSkillsCatalog();
    const second = loadSkillsCatalog();
    expect(second).toBe(first);
    answer();
    await expect(first).resolves.toEqual(SKILLS);
    expect(requests).toBe(1);
  });
});
