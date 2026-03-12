const menuToggle = document.getElementById("menuToggle");
const body = document.body;
const sidebarNav = document.querySelector(".sidebar nav");
const navSearch = document.getElementById("navSearch");

const closeSidebar = () => {
  body.classList.remove("sidebar-open");
  menuToggle?.setAttribute("aria-expanded", "false");
};

if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    const expanded = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!expanded));
    body.classList.toggle("sidebar-open", !expanded);
  });
}

document.querySelectorAll(".code-example").forEach((example) => {
  const tabs = example.querySelectorAll(".lang-tab");
  const blocks = example.querySelectorAll(".code-block");
  const copyBtn = example.querySelector(".copy-btn");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const selected = tab.dataset.lang;
      tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
      blocks.forEach((block) =>
        block.classList.toggle("is-active", block.dataset.lang === selected),
      );
    });
  });

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const activeCode =
        example.querySelector(".code-block.is-active code") ||
        example.querySelector(".code-block code");

      if (!activeCode) {
        return;
      }

      try {
        await navigator.clipboard.writeText(activeCode.textContent);
        copyBtn.classList.add("is-copied");
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.classList.remove("is-copied");
          copyBtn.textContent = "Copy";
        }, 1200);
      } catch {
        copyBtn.textContent = "Unavailable";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1200);
      }
    });
  }
});

if (sidebarNav && navSearch) {
  const groupTitles = [...sidebarNav.querySelectorAll(".nav-group-title")];
  const links = [...sidebarNav.querySelectorAll(".nav-link")];
  const isGroupTitle = (node) => node?.classList?.contains("nav-group-title");

  const applyNavFilter = () => {
    const query = navSearch.value.trim().toLowerCase();

    links.forEach((link) => {
      const isMatch = !query || link.textContent.toLowerCase().includes(query);
      link.classList.toggle("is-hidden", !isMatch);
    });

    groupTitles.forEach((title) => {
      let hasVisibleLink = false;
      let sibling = title.nextElementSibling;

      while (sibling && !isGroupTitle(sibling)) {
        if (sibling.classList.contains("nav-link") && !sibling.classList.contains("is-hidden")) {
          hasVisibleLink = true;
        }
        sibling = sibling.nextElementSibling;
      }

      title.classList.toggle("is-hidden", !hasVisibleLink);
    });
  };

  navSearch.addEventListener("input", applyNavFilter);
  applyNavFilter();
}

const navLinks = [...document.querySelectorAll(".nav-link"), ...document.querySelectorAll(".toc-link")];
const sectionById = new Map(
  [...document.querySelectorAll("main .section[id]")].map((section) => [section.id, section]),
);

const setActiveLink = (id) => {
  navLinks.forEach((link) => {
    const isMatch = link.getAttribute("href") === `#${id}`;
    link.classList.toggle("is-active", isMatch);
  });
};

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

    if (visible) {
      setActiveLink(visible.target.id);
    }
  },
  {
    rootMargin: "-24% 0px -62% 0px",
    threshold: [0.2, 0.4, 0.7],
  },
);

sectionById.forEach((section) => observer.observe(section));
if (window.location.hash) {
  setActiveLink(window.location.hash.slice(1));
}

document.querySelectorAll("a[href^='#']").forEach((anchor) => {
  anchor.addEventListener("click", () => {
    if (window.innerWidth <= 900) {
      closeSidebar();
    }
  });
});
