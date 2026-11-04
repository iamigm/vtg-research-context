/* Browser history controls shared by family, detail and comparison views. */
(() => {
  function refresh() {
    const navigation = window.navigation;
    document.querySelectorAll('[data-history-back]').forEach(button => {
      button.disabled = navigation ? !navigation.canGoBack : history.length <= 1;
    });
    document.querySelectorAll('[data-history-forward]').forEach(button => {
      // Older browsers do not expose the forward stack; history.forward() is safe at its end.
      button.disabled = navigation ? !navigation.canGoForward : false;
    });
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-history-back], [data-history-forward]');
    if (!button || button.disabled) return;
    if (button.hasAttribute('data-history-back')) history.back(); else history.forward();
  });
  for (const method of ['pushState', 'replaceState']) {
    const original = history[method];
    history[method] = function(...args) {
      const result = original.apply(this, args);
      refresh();
      return result;
    };
  }
  window.navigation?.addEventListener('currententrychange', refresh);
  window.addEventListener('popstate', refresh);
  window.addEventListener('pageshow', refresh);
  refresh();
})();
