/* Apply the saved theme and document language before the body is parsed. */
(function () {
  try {
    var theme = localStorage.getItem("os-theme");
    if (theme !== "dark" && theme !== "light" && theme !== "system") theme = "system";
    var dark = theme === "dark" || (theme === "system" && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
  } catch (error) {}
  var lang = null;
  try {
    lang = localStorage.getItem("os-lang");
  } catch (error) {}
  try {
    if (lang !== "zh" && lang !== "en") {
      // The first of the browser's languages the UI has, in the user's order
      // (i18n/runtime.ts systemLang): English first with Chinese further down
      // is English. A list naming neither is English.
      var locales = navigator.languages && navigator.languages.length
        ? navigator.languages
        : [navigator.language || ""];
      lang = "en";
      for (var i = 0; i < locales.length; i++) {
        if (/^zh(?:[-_]|$)/i.test(locales[i])) {
          lang = "zh";
          break;
        }
        if (/^en(?:[-_]|$)/i.test(locales[i])) break;
      }
    }
    document.documentElement.lang = lang;
  } catch (error) {}
})();
