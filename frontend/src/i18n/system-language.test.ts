import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Following the browser's language: the `languagechange` wiring the module
 * installs when it loads, a pick that storage refused, and the classic
 * `theme-bootstrap.js`, which must set the same `<html lang>` the bundle will.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const BOOTSTRAP = join(HERE, "../../../openai4s/server/webui/theme-bootstrap.js");

const original = {
  localStorage: Object.getOwnPropertyDescriptor(globalThis, "localStorage"),
  navigator: Object.getOwnPropertyDescriptor(globalThis, "navigator"),
  window: Object.getOwnPropertyDescriptor(globalThis, "window"),
};

function browser(languages: string[], language = languages[0] ?? ""): void {
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { languages, language },
  });
}

function storage(values: Record<string, string> = {}): void {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => values[k] ?? null,
      setItem: (k: string, v: string) => {
        values[k] = v;
      },
      removeItem: (k: string) => {
        delete values[k];
      },
    },
  });
}

function blockedStorage(): void {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new Error("SecurityError: site data is blocked");
    },
  });
}

afterEach(() => {
  for (const [name, descriptor] of Object.entries(original)) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete (globalThis as Record<string, unknown>)[name];
  }
  vi.resetModules();
});

async function freshRuntime(): Promise<typeof import("./runtime")> {
  vi.resetModules();
  return import("./runtime");
}

describe("following the browser's language", () => {
  it("loading the module listens for languagechange and follows it", async () => {
    const listeners: Record<string, () => void> = {};
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        addEventListener: (type: string, listener: () => void) => {
          listeners[type] = listener;
        },
      },
    });
    storage();
    browser(["zh-CN", "en"]);
    const runtime = await freshRuntime();
    await runtime.i18nReady();
    expect(runtime.LANG).toBe("zh");
    expect(listeners.languagechange).toBeTypeOf("function");

    browser(["en-US", "zh-CN"]);
    listeners.languagechange!();

    await vi.waitFor(() => expect(runtime.t("theme.toggle")).toBe("Toggle theme"));
    expect(runtime.LANG).toBe("en");
  });

  it("a pick that storage refused still counts for this page", async () => {
    blockedStorage();
    browser(["zh-CN"]);
    const runtime = await freshRuntime();
    await runtime.i18nReady();
    expect(runtime.langPreference()).toBe("system");

    await runtime.setLang("en");
    expect(runtime.langPreference()).toBe("en");

    browser(["zh-CN"]);
    await runtime.syncSystemLanguage();
    expect(runtime.LANG).toBe("en");

    await runtime.setLang("system");
    expect(runtime.langPreference()).toBe("system");
    expect(runtime.LANG).toBe("zh");
  });
});

describe("theme-bootstrap.js sets the language the bundle will use", () => {
  function bootstrapLang(
    languages: string[],
    options: { saved?: string; language?: string; blocked?: boolean } = {},
  ): string {
    const html = { lang: "zh", style: {}, setAttribute: () => undefined };
    const context = {
      navigator: { languages, language: options.language ?? languages[0] ?? "" },
      document: { documentElement: html },
      window: {},
      matchMedia: () => ({ matches: false }),
      get localStorage() {
        if (options.blocked) throw new Error("SecurityError: site data is blocked");
        return { getItem: (k: string) => (k === "os-lang" ? options.saved ?? null : null) };
      },
    };
    runInNewContext(readFileSync(BOOTSTRAP, "utf8"), context);
    return html.lang;
  }

  const LISTS: string[][] = [
    ["en-US", "en", "zh-CN"],
    ["zh-TW", "en-US"],
    ["fr-FR", "zh-CN", "en"],
    ["zh_Hans_CN"],
    ["ja-JP"],
    ["EN-gb", "zh"],
  ];

  it.each(LISTS)("agrees with systemLang for %j", async (...languages: string[]) => {
    storage();
    browser(languages);
    const { systemLang } = await import("./runtime");
    expect(bootstrapLang(languages)).toBe(systemLang());
  });

  it("falls back to navigator.language, honours a saved pick, survives blocked storage", () => {
    expect(bootstrapLang([], { language: "zh-CN" })).toBe("zh");
    expect(bootstrapLang(["en-US"], { saved: "zh" })).toBe("zh");
    expect(bootstrapLang(["en-US"], { blocked: true })).toBe("en");
  });
});
