"use strict";

const copyButton = document.querySelector("#copy-command");
const copyStatus = document.querySelector("#copy-status");
if (copyButton && navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    copyButton.textContent = "Copying…";
    copyButton.disabled = true;
    let timeout;
    try {
      await Promise.race([
        navigator.clipboard.writeText(
          document.querySelector("#install-command").textContent,
        ),
        new Promise((_, reject) => {
          timeout = setTimeout(
            () => reject(new Error("Clipboard unavailable")),
            1500,
          );
        }),
      ]);
      copyButton.textContent = "Copied";
      copyStatus.textContent = "Installation commands copied to clipboard.";
    } catch {
      copyButton.textContent = "Select text";
      copyStatus.textContent =
        "Clipboard unavailable. Select and copy the commands below.";
      const range = document.createRange();
      range.selectNodeContents(document.querySelector("#install-command"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    } finally {
      clearTimeout(timeout);
      copyButton.disabled = false;
    }
  });
}

if ("IntersectionObserver" in window) {
  const links = [...document.querySelectorAll("nav a")];
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const link of links) {
          if (link.hash === `#${entry.target.id}`)
            link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        }
      }
    },
    { rootMargin: "-10% 0px -60% 0px" },
  );
  for (const link of links) observer.observe(document.querySelector(link.hash));
}
