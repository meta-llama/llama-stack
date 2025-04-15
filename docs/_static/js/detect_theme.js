document.addEventListener("DOMContentLoaded", function () {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const htmlElement = document.documentElement;
  if (prefersDark) {
    htmlElement.setAttribute("data-theme", "dark");
  } else {
    htmlElement.setAttribute("data-theme", "light");
  }
});
