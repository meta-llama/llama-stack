document.addEventListener("DOMContentLoaded", function () {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const htmlElement = document.documentElement;

  // Check if theme is saved in localStorage
  const savedTheme = localStorage.getItem("sphinx-rtd-theme");

  if (savedTheme) {
    // Use the saved theme preference
    htmlElement.setAttribute("data-theme", savedTheme);
    document.body.classList.toggle("dark", savedTheme === "dark");
  } else {
    // Fall back to system preference
    const theme = prefersDark ? "dark" : "light";
    htmlElement.setAttribute("data-theme", theme);
    document.body.classList.toggle("dark", theme === "dark");
    // Save initial preference
    localStorage.setItem("sphinx-rtd-theme", theme);
  }

  // Listen for theme changes from the existing toggle
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === "data-theme") {
        const currentTheme = htmlElement.getAttribute("data-theme");
        localStorage.setItem("sphinx-rtd-theme", currentTheme);
      }
    });
  });

  observer.observe(htmlElement, { attributes: true });
});
